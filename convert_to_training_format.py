#!/usr/bin/env python3
"""
Convert our HF-format dataset (with 'messages' and structured content)
to the Qwen-VL training format (with 'conversations' and <video> tags).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def convert_messages_to_conversations(messages: List[Dict]) -> List[Dict]:
    """Convert HF messages format to Qwen-VL conversations format."""
    conversations = []

    for message in messages:
        role = message["role"]

        # Skip system messages or merge into first user message
        if role == "system":
            continue

        # Map role to Qwen-VL format
        from_role = "human" if role == "user" else "gpt"

        # Extract text and media from content
        text_parts = []
        has_video = False

        for item in message["content"]:
            if item["type"] == "text":
                text_parts.append(item["text"])
            elif item["type"] == "video":
                # Insert <video> placeholder where video appears
                text_parts.append("<video>")
                has_video = True

        # Join all text parts
        value = "\n".join(text_parts)

        conversations.append({
            "from": from_role,
            "value": value
        })

    return conversations


def convert_sample(sample: Dict, base_path: Path) -> Dict:
    """Convert a single sample to training format."""
    # Convert messages to conversations
    conversations = convert_messages_to_conversations(sample["messages"])

    # Get video path (make relative to dataset root)
    video_path = sample["videos"]

    # Create training-format sample
    training_sample = {
        "video": video_path,
        "conversations": conversations
    }

    return training_sample


def convert_dataset(
    input_path: Path,
    output_path: Path,
    base_path: Path = Path(".")
) -> None:
    """Convert entire dataset from HF format to training format."""

    # Load HF dataset
    with input_path.open() as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {input_path}")

    # Convert each sample
    training_samples = []
    for sample in samples:
        try:
            training_sample = convert_sample(sample, base_path)
            training_samples.append(training_sample)
        except Exception as e:
            print(f"Warning: Failed to convert sample {sample.get('segment_uuid', 'unknown')}: {e}")

    # Save training-format dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(training_samples, f, indent=2)

    print(f"Saved {len(training_samples)} converted samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF dataset to Qwen-VL training format"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("qwen_video_dataset"),
        help="Directory containing the HF dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen_video_dataset_training"),
        help="Directory for converted training dataset"
    )

    args = parser.parse_args()

    # Convert train and test splits
    for split in ["train", "test"]:
        input_path = args.dataset_dir / f"{split}_examples.json"
        output_path = args.output_dir / f"{split}.json"

        if input_path.exists():
            print(f"\n=== Converting {split} split ===")
            convert_dataset(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found, skipping {split} split")


if __name__ == "__main__":
    main()
