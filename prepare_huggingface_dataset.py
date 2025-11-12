#!/usr/bin/env python3
"""
Convert extracted 1-second video segments + ego-motion metadata into a
Hugging Face dataset tailored for Qwen3-VL LoRA training.

Each example pairs a 30 FPS video clip with **exactly 30** (velocity, curvature)
pairs covering the following second, formatted as [(xx.x, 0.xxx), ...].
"""

import argparse
import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MICROSECONDS_IN_SECOND = 1_000_000


@dataclass
class DatasetPaths:
    segments_path: Path
    ego_motion_dir: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hugging Face dataset with video -> motion targets pairs."
    )
    parser.add_argument(
        "--segments-path",
        type=Path,
        default=Path("qwen_video_segments/metadata/segments.json"),
        help="Path to segments metadata JSON produced by prepare_video_segments.py",
    )
    parser.add_argument(
        "--ego-motion-dir",
        type=Path,
        default=Path("physical_ai_data/ego_motion_chunk_0000"),
        help="Directory containing *.egomotion.parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen_video_dataset"),
        help="Target directory for the Hugging Face dataset (will be overwritten)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to reserve for the test split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for shuffling before the train/test split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of segments to convert (useful for smoke tests)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for both the video segment and the predicted horizon",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration in seconds for the video clip and prediction horizon",
    )
    parser.add_argument(
        "--examples-preview",
        type=int,
        default=5,
        help="Number of sample rows to dump into train_examples.json/test_examples.json",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are an autonomy stack assistant that must predict the next second "
            "of ego motion as velocity/curvature pairs."
        ),
        help="System prompt injected into every conversation",
    )
    parser.add_argument(
        "--user-instruction",
        type=str,
        default=(
            "Given this 1-second, {fps} FPS driving clip, output exactly {num_pairs} "
            "pairs of (velocity in m/s, curvature) for the next second as "
            "[(xx.x, 0.xxx), ...]. No prose, just the list."
        ),
        help="Template for the user turn. Supports {fps} and {num_pairs} placeholders.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite the dataset output directory if it already exists.",
    )
    return parser.parse_args()


def ensure_velocity_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    if "velocity_magnitude" not in df.columns:
        df = df.copy()
        df["velocity_magnitude"] = np.sqrt(
            df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2
        )
    return df


def load_ego_motion(ego_motion_dir: Path, clip_uuid: str) -> pd.DataFrame:
    ego_path = ego_motion_dir / f"{clip_uuid}.egomotion.parquet"
    if not ego_path.exists():
        raise FileNotFoundError(f"Ego motion parquet missing: {ego_path}")
    df = pd.read_parquet(ego_path)
    df = ensure_velocity_magnitude(df)
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_motion_targets(
    ego_df: pd.DataFrame,
    target_start_time_us: int,
    fps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return 30 velocity/curvature values covering the next second."""
    target_end_time = target_start_time_us + MICROSECONDS_IN_SECOND
    window = ego_df[
        (ego_df["timestamp"] >= target_start_time_us)
        & (ego_df["timestamp"] <= target_end_time)
    ]

    if window.empty:
        logger.warning(
            "No ego motion samples found between %s and %s. Returning zeros.",
            target_start_time_us,
            target_end_time,
        )
        return np.zeros(fps), np.zeros(fps)

    timestamps = (
        window["timestamp"].to_numpy() - target_start_time_us
    ) / MICROSECONDS_IN_SECOND
    velocities = window["velocity_magnitude"].to_numpy()
    curvatures = window["curvature"].to_numpy()

    # Remove duplicate timestamps to keep np.interp happy.
    unique_idx = np.unique(timestamps, return_index=True)[1]
    timestamps = timestamps[unique_idx]
    velocities = velocities[unique_idx]
    curvatures = curvatures[unique_idx]

    sample_points = (np.arange(fps) + 0.5) / fps  # center of each frame interval

    velocity_series = np.interp(
        sample_points, timestamps, velocities, left=velocities[0], right=velocities[-1]
    )
    curvature_series = np.interp(
        sample_points, timestamps, curvatures, left=curvatures[0], right=curvatures[-1]
    )

    return velocity_series, curvature_series


def format_velocity(value: float) -> str:
    return f"{value:.1f}".zfill(4)


def format_curvature(value: float) -> str:
    return f"{abs(value):.3f}"


def build_pair_strings(
    velocities: Sequence[float], curvatures: Sequence[float]
) -> Tuple[str, List[List[float]]]:
    formatted_pairs = []
    numeric_pairs: List[List[float]] = []
    for velocity, curvature in zip(velocities, curvatures):
        formatted_velocity = format_velocity(velocity)
        formatted_curvature = format_curvature(curvature)
        formatted_pairs.append(f"({formatted_velocity}, {formatted_curvature})")
        numeric_pairs.append([float(f"{velocity:.5f}"), float(f"{curvature:.6f}")])
    pair_text = "[" + ", ".join(formatted_pairs) + "]"
    return pair_text, numeric_pairs


def build_messages(
    video_path: str,
    fps: int,
    num_frames: int,
    system_prompt: str,
    user_instruction_template: str,
    assistant_answer: str,
) -> List[Dict]:
    user_instruction = user_instruction_template.format(fps=fps, num_pairs=num_frames)
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instruction},
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "num_frames": num_frames,
                    "duration_seconds": num_frames / fps,
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_answer}]},
    ]


def split_dataset(
    records: List[Dict],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    random.Random(seed).shuffle(records)
    if not records:
        return [], []
    split_index = int(len(records) * (1 - test_ratio))
    split_index = (
        max(1, min(split_index, len(records) - 1)) if len(records) > 1 else len(records)
    )
    return records[:split_index], records[split_index:]


def save_examples_preview(
    examples: List[Dict],
    path: Path,
    limit: int,
) -> None:
    subset = examples[:limit]
    with path.open("w") as fp:
        json.dump(subset, fp, indent=2)


def prepare_dataset(
    *,
    segments_path: Path,
    ego_motion_dir: Path,
    output_dir: Path,
    test_ratio: float = 0.1,
    seed: int = 7,
    max_samples: Optional[int] = None,
    fps: int = 30,
    duration: float = 1.0,
    examples_preview: int = 5,
    system_prompt: str,
    user_instruction: str,
    overwrite_output: bool = False,
) -> DatasetDict:
    paths = DatasetPaths(
        segments_path=segments_path,
        ego_motion_dir=ego_motion_dir,
        output_dir=output_dir,
    )

    if not paths.segments_path.exists():
        raise FileNotFoundError(f"Segments metadata not found: {paths.segments_path}")
    if not paths.ego_motion_dir.exists():
        raise FileNotFoundError(
            f"Ego motion directory not found: {paths.ego_motion_dir}"
        )

    with paths.segments_path.open() as fp:
        segments = json.load(fp)

    if max_samples:
        segments = segments[:max_samples]

    logger.info("Loaded %d segment metadata entries", len(segments))

    ego_cache: Dict[str, pd.DataFrame] = {}
    all_records: List[Dict] = []

    for segment in segments:
        clip_uuid = segment["clip_uuid"]
        segment_uuid = segment["segment_uuid"]
        start_second = segment["start_second"]
        video_path = segment["video_path"]

        try:
            if clip_uuid not in ego_cache:
                ego_cache[clip_uuid] = load_ego_motion(paths.ego_motion_dir, clip_uuid)

            target_start_us = int((start_second + duration) * MICROSECONDS_IN_SECOND)
            velocities, curvatures = resample_motion_targets(
                ego_cache[clip_uuid], target_start_us, fps
            )
            if len(velocities) != fps:
                raise ValueError(
                    f"Expected {fps} samples, got {len(velocities)} for {segment_uuid}"
                )
            pair_text, numeric_pairs = build_pair_strings(velocities, curvatures)

            record = {
                "messages": build_messages(
                    video_path=video_path,
                    fps=fps,
                    num_frames=int(fps * duration),
                    system_prompt=system_prompt,
                    user_instruction_template=user_instruction,
                    assistant_answer=pair_text,
                ),
                "videos": video_path,
                "segment_uuid": segment_uuid,
                "clip_uuid": clip_uuid,
                "start_second": start_second,
                "video_fps": fps,
                "video_num_frames": int(fps * duration),
                "targets_text": pair_text,
                "targets": numeric_pairs,
            }
            all_records.append(record)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process segment %s: %s", segment_uuid, exc)

    if not all_records:
        raise RuntimeError("No records were generated. Check logs for earlier errors.")

    train_records, test_records = split_dataset(all_records, test_ratio, seed)

    logger.info(
        "Prepared %d train and %d test samples (%.2f%% test split)",
        len(train_records),
        len(test_records),
        test_ratio * 100,
    )

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "test": Dataset.from_list(test_records),
        }
    )

    if paths.output_dir.exists():
        if overwrite_output:
            shutil.rmtree(paths.output_dir)
        else:
            raise FileExistsError(
                f"Output directory {paths.output_dir} already exists. "
                "Pass --overwrite-output to replace it."
            )

    dataset_dict.save_to_disk(str(paths.output_dir))
    (paths.output_dir / "dataset_dict.json").write_text(
        json.dumps({"splits": ["train", "test"]})
    )

    save_examples_preview(
        train_records, paths.output_dir / "train_examples.json", examples_preview
    )
    save_examples_preview(
        test_records, paths.output_dir / "test_examples.json", examples_preview
    )

    logger.info("Saved dataset to %s", paths.output_dir)
    return dataset_dict


def main() -> None:
    args = parse_args()
    prepare_dataset(
        segments_path=args.segments_path,
        ego_motion_dir=args.ego_motion_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        fps=args.fps,
        duration=args.duration,
        examples_preview=args.examples_preview,
        system_prompt=args.system_prompt,
        user_instruction=args.user_instruction,
        overwrite_output=args.overwrite_output,
    )


if __name__ == "__main__":
    main()
