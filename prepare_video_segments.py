#!/usr/bin/env python3
"""
Script to prepare 1-second video segments from autonomous vehicle dataset
for Qwen-VL LoRA training.

Extracts 1-second clips from 20-second source videos and pairs them with
corresponding ego motion targets for the next second.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoSegmentExtractor:
    """Extracts 1-second video segments with motion targets for training."""

    def __init__(self, data_dir: str, output_dir: str, fps: int = 30):
        """
        Initialize the extractor.

        Args:
            data_dir: Directory containing the raw dataset
            output_dir: Directory to save extracted segments
            fps: Target frames per second for segments
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.segment_duration = 1.0  # 1 second segments
        self.target_duration = 1.0  # 1 second prediction target

        # Create output directories
        self.video_dir = self.output_dir / "video_segments"
        self.metadata_dir = self.output_dir / "metadata"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def load_clip_data(self, clip_uuid: str) -> Tuple[cv2.VideoCapture, pd.DataFrame]:
        """
        Load video and ego motion data for a clip.

        Args:
            clip_uuid: UUID of the clip to load

        Returns:
            Tuple of (video_capture, ego_motion_df)
        """
        # Load video
        video_path = (
            self.data_dir
            / f"camera_chunk_0000/{clip_uuid}.camera_front_wide_120fov.mp4"
        )
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Load ego motion data
        ego_path = (
            self.data_dir / f"ego_motion_chunk_0000/{clip_uuid}.egomotion.parquet"
        )
        if not ego_path.exists():
            raise FileNotFoundError(f"Ego motion not found: {ego_path}")

        ego_df = pd.read_parquet(ego_path)

        # Calculate velocity magnitude if not present
        if "velocity_magnitude" not in ego_df.columns:
            ego_df["velocity_magnitude"] = np.sqrt(
                ego_df["vx"] ** 2 + ego_df["vy"] ** 2 + ego_df["vz"] ** 2
            )

        return cap, ego_df

    def load_camera_timestamps(self, clip_uuid: str) -> pd.DataFrame:
        """
        Load camera timestamp data for a clip.

        Args:
            clip_uuid: UUID of the clip

        Returns:
            DataFrame with timestamp data
        """
        ts_path = (
            self.data_dir
            / f"camera_chunk_0000/{clip_uuid}.camera_front_wide_120fov.timestamps.parquet"
        )
        if not ts_path.exists():
            raise FileNotFoundError(f"Timestamps not found: {ts_path}")

        return pd.read_parquet(ts_path)

    def extract_video_segment(
        self, cap: cv2.VideoCapture, start_second: float, segment_uuid: str
    ) -> str:
        """
        Extract a 1-second video segment and save it.

        Args:
            cap: OpenCV video capture object
            start_second: Start time in seconds
            segment_uuid: Unique identifier for the segment

        Returns:
            Path to the saved video segment
        """
        # Calculate frame indices
        start_frame = int(start_second * self.fps)
        end_frame = int((start_second + self.segment_duration) * self.fps)

        # Set video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        output_path = self.video_dir / f"{segment_uuid}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))

        # Extract frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_extracted = 0
        for frame_idx in range(
            start_frame, min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                frames_extracted += 1
            else:
                break

        out.release()

        if frames_extracted < self.fps:
            logger.warning(f"Segment {segment_uuid} only has {frames_extracted} frames")

        return str(output_path)

    def calculate_motion_targets(
        self, ego_df: pd.DataFrame, target_start_time: int
    ) -> Tuple[float, float]:
        """
        Calculate average velocity and curvature for target time window.

        Args:
            ego_df: Ego motion DataFrame
            target_start_time: Start time of target window in microseconds

        Returns:
            Tuple of (average_velocity, average_curvature)
        """
        # Define target time window (1 second)
        target_end_time = target_start_time + int(
            self.target_duration * 1_000_000
        )  # Convert to microseconds

        # Filter ego motion data for target window
        target_data = ego_df[
            (ego_df["timestamp"] > target_start_time)
            & (ego_df["timestamp"] <= target_end_time)
        ]

        if len(target_data) == 0:
            logger.warning(
                f"No ego motion data found for target window {target_start_time}-{target_end_time}"
            )
            return 0.0, 0.0

        # Calculate averages
        avg_velocity = float(target_data["velocity_magnitude"].mean())
        avg_curvature = float(target_data["curvature"].mean())

        return avg_velocity, avg_curvature

    def format_motion_targets(
        self, velocity: float, curvature: float
    ) -> Tuple[str, str]:
        """
        Format motion targets according to specified output format.

        Args:
            velocity: Average velocity in m/s
            curvature: Average curvature

        Returns:
            Tuple of (formatted_velocity, formatted_curvature)
        """
        # Format velocity as xx.x (one decimal place)
        formatted_velocity = f"{velocity:.1f}"

        # Format curvature as 0.xxx (three decimal places)
        formatted_curvature = f"{abs(curvature):.3f}"

        return formatted_velocity, formatted_curvature

    def process_clip(self, clip_uuid: str) -> List[Dict]:
        """
        Process a single clip and extract all possible 1-second segments.

        Args:
            clip_uuid: UUID of the clip to process

        Returns:
            List of segment metadata dictionaries
        """
        try:
            # Load data
            cap, ego_df = self.load_clip_data(clip_uuid)
            ts_df = self.load_camera_timestamps(clip_uuid)

            segments = []

            # Create 1-second segments
            # We can create 19 segments from a 20-second clip (sliding window)
            for start_second in range(0, 19):  # 0 to 18 inclusive
                # Generate unique segment ID
                segment_uuid = f"{clip_uuid}_seg_{start_second:02d}"

                # Extract video segment
                video_path = self.extract_video_segment(cap, start_second, segment_uuid)

                # Get target time (next second)
                target_start_time = int(
                    (start_second + 1) * 1_000_000
                )  # Convert to microseconds

                # Calculate motion targets
                avg_velocity, avg_curvature = self.calculate_motion_targets(
                    ego_df, target_start_time
                )

                # Format targets
                formatted_velocity, formatted_curvature = self.format_motion_targets(
                    avg_velocity, avg_curvature
                )

                # Create segment metadata
                segment_metadata = {
                    "segment_uuid": segment_uuid,
                    "clip_uuid": clip_uuid,
                    "start_second": start_second,
                    "video_path": video_path,
                    "target_velocity": formatted_velocity,
                    "target_curvature": formatted_curvature,
                    "target_velocity_raw": avg_velocity,
                    "target_curvature_raw": avg_curvature,
                }

                segments.append(segment_metadata)

            cap.release()
            return segments

        except Exception as e:
            logger.error(f"Failed to process clip {clip_uuid}: {e}")
            return []

    def process_all_clips(self, max_clips: int = None) -> None:
        """
        Process all clips in the dataset.

        Args:
            max_clips: Maximum number of clips to process (for testing)
        """
        # Find all available clips
        ego_dir = self.data_dir / "ego_motion_chunk_0000"
        if not ego_dir.exists():
            raise FileNotFoundError(f"Ego motion directory not found: {ego_dir}")

        clip_files = list(ego_dir.glob("*.egomotion.parquet"))
        clip_uuids = [f.stem.replace(".egomotion", "") for f in clip_files]

        if max_clips:
            clip_uuids = clip_uuids[:max_clips]

        logger.info(f"Processing {len(clip_uuids)} clips")

        all_segments = []

        for clip_uuid in tqdm(clip_uuids, desc="Processing clips"):
            segments = self.process_clip(clip_uuid)
            all_segments.extend(segments)

        # Save segment metadata
        metadata_path = self.metadata_dir / "segments.json"
        with open(metadata_path, "w") as f:
            json.dump(all_segments, f, indent=2)

        logger.info(f"Extracted {len(all_segments)} segments total")
        logger.info(f"Video segments saved to: {self.video_dir}")
        logger.info(f"Metadata saved to: {metadata_path}")

        # Print summary statistics
        self.print_summary_statistics(all_segments)

    def print_summary_statistics(self, segments: List[Dict]) -> None:
        """Print summary statistics about extracted segments."""
        if not segments:
            return

        velocities = [s["target_velocity_raw"] for s in segments]
        curvatures = [s["target_curvature_raw"] for s in segments]

        print("\n" + "=" * 50)
        print("SEGMENT EXTRACTION SUMMARY")
        print("=" * 50)
        print(f"Total segments extracted: {len(segments)}")
        print(f"Total clips processed: {len(set(s['clip_uuid'] for s in segments))}")
        print(f"Segments per clip: 19 (from 20-second clips)")
        print()
        print("Velocity statistics (m/s):")
        print(f"  Mean: {np.mean(velocities):.2f}")
        print(f"  Std:  {np.std(velocities):.2f}")
        print(f"  Min:  {np.min(velocities):.2f}")
        print(f"  Max:  {np.max(velocities):.2f}")
        print()
        print("Curvature statistics:")
        print(f"  Mean: {np.mean(curvatures):.4f}")
        print(f"  Std:  {np.std(curvatures):.4f}")
        print(f"  Min:  {np.min(curvatures):.4f}")
        print(f"  Max:  {np.max(curvatures):.4f}")
        print("=" * 50)


def main():
    """Main function to run video segment extraction."""
    parser = argparse.ArgumentParser(
        description="Extract 1-second video segments from autonomous vehicle dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="physical_ai_data",
        help="Directory containing the raw dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen_video_segments",
        help="Directory to save extracted segments",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Maximum number of clips to process (for testing)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target FPS for extracted segments"
    )

    args = parser.parse_args()

    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Create extractor and process clips
    extractor = VideoSegmentExtractor(
        data_dir=args.input_dir, output_dir=args.output_dir, fps=args.fps
    )

    extractor.process_all_clips(max_clips=args.max_clips)


if __name__ == "__main__":
    main()
