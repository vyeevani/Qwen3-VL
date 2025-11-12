#!/usr/bin/env python3
"""
Evaluate the fine-tuned Qwen3-VL model on autonomous driving video prediction.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def parse_motion_pairs(text: str) -> List[Tuple[float, float]]:
    """Parse velocity/curvature pairs from model output.

    Expected format: [(xx.x, 0.xxx), (xx.x, 0.xxx), ...]
    """
    # Remove any prose and extract just the list
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find list in output: {text}")

    pairs_text = match.group(1)

    # Extract individual pairs: (xx.x, 0.xxx)
    pair_pattern = r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)'
    matches = re.findall(pair_pattern, pairs_text)

    if not matches:
        raise ValueError(f"Could not parse pairs from: {pairs_text}")

    pairs = [(float(v), float(c)) for v, c in matches]
    return pairs


def calculate_metrics(predicted: List[Tuple[float, float]],
                     ground_truth: List[Tuple[float, float]]) -> dict:
    """Calculate evaluation metrics."""
    # Ensure same length
    min_len = min(len(predicted), len(ground_truth))
    predicted = predicted[:min_len]
    ground_truth = ground_truth[:min_len]

    # Separate velocity and curvature
    pred_v = [p[0] for p in predicted]
    pred_c = [p[1] for p in predicted]
    gt_v = [p[0] for p in ground_truth]
    gt_c = [p[1] for p in ground_truth]

    # Calculate MAE and MSE
    import numpy as np

    velocity_mae = np.mean(np.abs(np.array(pred_v) - np.array(gt_v)))
    velocity_mse = np.mean((np.array(pred_v) - np.array(gt_v)) ** 2)
    velocity_rmse = np.sqrt(velocity_mse)

    curvature_mae = np.mean(np.abs(np.array(pred_c) - np.array(gt_c)))
    curvature_mse = np.mean((np.array(pred_c) - np.array(gt_c)) ** 2)
    curvature_rmse = np.sqrt(curvature_mse)

    return {
        "velocity": {
            "mae": velocity_mae,
            "mse": velocity_mse,
            "rmse": velocity_rmse,
            "mean_predicted": np.mean(pred_v),
            "mean_ground_truth": np.mean(gt_v),
        },
        "curvature": {
            "mae": curvature_mae,
            "mse": curvature_mse,
            "rmse": curvature_rmse,
            "mean_predicted": np.mean(pred_c),
            "mean_ground_truth": np.mean(gt_c),
        },
        "num_pairs": min_len,
    }


def evaluate_sample(model, processor, sample: dict, device: str = "cuda") -> dict:
    """Evaluate a single test sample."""

    # Get video path and ground truth
    video_path = sample["video"]
    conversations = sample["conversations"]

    # Find the ground truth answer
    ground_truth_text = None
    for conv in conversations:
        if conv["from"] == "gpt":
            ground_truth_text = conv["value"]
            break

    if not ground_truth_text:
        raise ValueError("No ground truth found in conversations")

    # Build messages for inference (without assistant response)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given this 1-second, 30 FPS driving clip, output exactly 30 pairs of (velocity in m/s, curvature) for the next second as [(xx.x, 0.xxx), ...]. No prose, just the list."
                },
                {"type": "video", "video": video_path, "fps": 30}
            ]
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate prediction
    print(f"Running inference on {video_path}...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Deterministic for evaluation
        )

    # Decode prediction
    prediction_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(f"\n{'='*60}")
    print(f"Video: {video_path}")
    print(f"{'='*60}")
    print(f"\nPredicted:\n{prediction_text}")
    print(f"\nGround Truth:\n{ground_truth_text}")
    print(f"{'='*60}\n")

    # Parse pairs
    try:
        predicted_pairs = parse_motion_pairs(prediction_text)
        ground_truth_pairs = parse_motion_pairs(ground_truth_text)

        # Calculate metrics
        metrics = calculate_metrics(predicted_pairs, ground_truth_pairs)

        return {
            "video": video_path,
            "predicted_text": prediction_text,
            "ground_truth_text": ground_truth_text,
            "predicted_pairs": predicted_pairs,
            "ground_truth_pairs": ground_truth_pairs,
            "metrics": metrics,
            "success": True,
        }
    except Exception as e:
        print(f"Error parsing output: {e}")
        return {
            "video": video_path,
            "predicted_text": prediction_text,
            "ground_truth_text": ground_truth_text,
            "error": str(e),
            "success": False,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base model path"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="./checkpoints/physical_ai_driving_lora",
        help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("qwen_video_dataset_training/test.json"),
        help="Path to test data JSON"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also evaluate base model without LoRA for comparison"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Qwen3-VL Autonomous Driving Evaluation")
    print(f"{'='*60}\n")

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data) as f:
        test_samples = json.load(f)
    print(f"Found {len(test_samples)} test sample(s)\n")

    # Load base model
    print(f"Loading base model: {args.base_model}...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2"
    )

    processor = AutoProcessor.from_pretrained(args.base_model)

    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    print("Model loaded successfully!\n")

    # Evaluate each sample
    results = []
    for i, sample in enumerate(test_samples):
        print(f"\n--- Evaluating Sample {i+1}/{len(test_samples)} ---\n")
        result = evaluate_sample(model, processor, sample, args.device)
        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}\n")

    successful_results = [r for r in results if r["success"]]

    if successful_results:
        # Aggregate metrics
        import numpy as np

        all_velocity_mae = [r["metrics"]["velocity"]["mae"] for r in successful_results]
        all_velocity_rmse = [r["metrics"]["velocity"]["rmse"] for r in successful_results]
        all_curvature_mae = [r["metrics"]["curvature"]["mae"] for r in successful_results]
        all_curvature_rmse = [r["metrics"]["curvature"]["rmse"] for r in successful_results]

        print(f"Samples evaluated: {len(successful_results)}/{len(test_samples)}")
        print(f"\nVelocity Metrics:")
        print(f"  MAE:  {np.mean(all_velocity_mae):.4f} m/s")
        print(f"  RMSE: {np.mean(all_velocity_rmse):.4f} m/s")

        print(f"\nCurvature Metrics:")
        print(f"  MAE:  {np.mean(all_curvature_mae):.6f}")
        print(f"  RMSE: {np.mean(all_curvature_rmse):.6f}")

        # Show detailed results for each sample
        for i, result in enumerate(successful_results):
            print(f"\n--- Sample {i+1} Details ---")
            print(f"Video: {Path(result['video']).name}")
            print(f"Velocity  - MAE: {result['metrics']['velocity']['mae']:.4f} m/s, "
                  f"RMSE: {result['metrics']['velocity']['rmse']:.4f} m/s")
            print(f"Curvature - MAE: {result['metrics']['curvature']['mae']:.6f}, "
                  f"RMSE: {result['metrics']['curvature']['rmse']:.6f}")
            print(f"Pairs compared: {result['metrics']['num_pairs']}")
    else:
        print("No successful evaluations!")

    print(f"\n{'='*60}\n")

    # Save results
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
