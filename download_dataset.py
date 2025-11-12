#!/usr/bin/env python3
"""
Download script for the PhysicalAI-Autonomous-Vehicles dataset.
Downloads front wide camera, ego motion, camera intrinsics, and extrinsics data for specified chunks.
Automatically unzips downloaded files for easy access.

Usage:
    uv run python download_dataset.py --chunk 0000  # Downloads first chunk (default)
    uv run python download_dataset.py --chunk 0001  # Downloads second chunk
    uv run python download_dataset.py --chunk 0000 0001 0002  # Downloads multiple chunks
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


def authenticate() -> bool:
    """
    Authenticate with Hugging Face.
    Returns True if authentication successful, False otherwise.
    """
    try:
        # Check if already logged in
        api = HfApi()
        whoami = api.whoami()
        print(f"Already authenticated as: {whoami['name']}")
        return True
    except Exception:
        print("Hugging Face authentication required.")
        print("Please run: uv run huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        return False


def get_chunk_files(chunk_ids: List[str]) -> List[tuple]:
    """
    Get the list of files to download for specified chunks.
    Returns list of (relative_path, local_filename) tuples.
    """
    files_to_download = []

    # Front wide camera data
    for chunk_id in chunk_ids:
        camera_path = f"camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{chunk_id}.zip"
        local_camera = f"camera_front_wide_120fov_chunk_{chunk_id}.zip"
        files_to_download.append((camera_path, local_camera))

    # Ego motion data
    for chunk_id in chunk_ids:
        ego_path = f"labels/egomotion/egomotion.chunk_{chunk_id}.zip"
        local_ego = f"ego_motion_chunk_{chunk_id}.zip"
        files_to_download.append((ego_path, local_ego))

    # Camera intrinsics data
    for chunk_id in chunk_ids:
        intrinsics_path = (
            f"calibration/camera_intrinsics/camera_intrinsics.chunk_{chunk_id}.parquet"
        )
        local_intrinsics = f"camera_intrinsics_chunk_{chunk_id}.parquet"
        files_to_download.append((intrinsics_path, local_intrinsics))

    # Sensor extrinsics data
    for chunk_id in chunk_ids:
        extrinsics_path = (
            f"calibration/sensor_extrinsics/sensor_extrinsics.chunk_{chunk_id}.parquet"
        )
        local_extrinsics = f"sensor_extrinsics_chunk_{chunk_id}.parquet"
        files_to_download.append((extrinsics_path, local_extrinsics))

    # Vehicle dimensions data
    for chunk_id in chunk_ids:
        vehicle_path = f"calibration/vehicle_dimensions/vehicle_dimensions.chunk_{chunk_id}.parquet"
        local_vehicle = f"vehicle_dimensions_chunk_{chunk_id}.parquet"
        files_to_download.append((vehicle_path, local_vehicle))

    return files_to_download


def unzip_file(zip_path: Path, extract_to: Path, show_progress: bool = True) -> bool:
    """
    Unzip a file to the specified directory.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to
        show_progress: Whether to show extraction progress

    Returns:
        True if successful, False otherwise
    """
    try:
        if show_progress:
            print(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            if show_progress:
                members = zf.namelist()
                pbar = tqdm(members, desc=f"Extracting {zip_path.name}", unit="file")
                for member in pbar:
                    zf.extract(member, extract_to)
                    pbar.set_postfix_str(f"Extracted: {member}")
            else:
                zf.extractall(extract_to)

        # Remove the zip file after successful extraction
        zip_path.unlink()
        if show_progress:
            print(f"  Removed zip file: {zip_path.name}")

        return True

    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def download_files(
    repo_id: str, files: List[tuple], output_dir: Path, show_progress: bool = True
) -> List[Path]:
    """
    Download files from Hugging Face repository and unzip zip files.

    Args:
        repo_id: Repository identifier (e.g., "nvidia/PhysicalAI-Autonomous-Vehicles")
        files: List of (relative_path, local_filename) tuples
        output_dir: Directory to save files
        show_progress: Whether to show progress bar

    Returns:
        List of successfully downloaded file/directory paths
    """
    downloaded_files = []

    if show_progress:
        pbar = tqdm(files, desc="Downloading files", unit="file")
    else:
        pbar = files

    for repo_path, local_name in pbar:
        try:
            # Handle different file types
            if local_name.endswith(".zip"):
                # Camera or ego motion zip files
                chunk_id = local_name.split("_chunk_")[1].replace(".zip", "")

                # Check if extracted content already exists
                if "camera_front_wide" in local_name:
                    extract_dir = output_dir / f"camera_chunk_{chunk_id}"
                elif "ego_motion" in local_name:
                    extract_dir = output_dir / f"ego_motion_chunk_{chunk_id}"
                else:
                    extract_dir = output_dir / f"chunk_{chunk_id}"

                if extract_dir.exists() and any(extract_dir.iterdir()):
                    if show_progress:
                        pbar.set_postfix_str(
                            f"Skipped (already extracted): {local_name}"
                        )
                    downloaded_files.append(extract_dir)
                    continue

                # Download file
                if show_progress:
                    pbar.set_postfix_str(f"Downloading: {local_name}")

                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

                # Move to final location
                local_path = output_dir / local_name
                actual_downloaded_path = Path(downloaded_path)
                if actual_downloaded_path != local_path:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    if actual_downloaded_path.exists():
                        actual_downloaded_path.rename(local_path)

                # Extract the zip file
                extract_dir.mkdir(exist_ok=True)
                if unzip_file(local_path, extract_dir, show_progress):
                    downloaded_files.append(extract_dir)
                else:
                    downloaded_files.append(local_path)

            elif local_name.endswith(".parquet"):
                # Parquet files (intrinsics, extrinsics, vehicle dimensions)
                local_path = output_dir / local_name

                # Check if file already exists
                if local_path.exists():
                    if show_progress:
                        pbar.set_postfix_str(f"Skipped (exists): {local_name}")
                    downloaded_files.append(local_path)
                    continue

                # Download file
                if show_progress:
                    pbar.set_postfix_str(f"Downloading: {local_name}")

                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

                # Move to final location
                actual_downloaded_path = Path(downloaded_path)
                if actual_downloaded_path != local_path:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    if actual_downloaded_path.exists():
                        actual_downloaded_path.rename(local_path)

                downloaded_files.append(local_path)

                if show_progress:
                    size_mb = local_path.stat().st_size / (1024**2)
                    pbar.set_postfix_str(f"Downloaded: {local_name} ({size_mb:.1f}MB)")

        except Exception as e:
            if show_progress:
                pbar.set_postfix_str(f"Failed: {local_name} - {str(e)[:50]}")
            print(f"Warning: Failed to download {repo_path}: {e}")

    # Clean up empty directories that might be created by hf_hub_download
    if show_progress:
        pbar.set_postfix_str("Cleaning up...")

    for root, dirs, files_in_dir in os.walk(output_dir, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
            except OSError:
                pass  # Directory not empty or other error

    return downloaded_files


def validate_chunk_range(chunk_ids: List[str]) -> bool:
    """
    Validate that chunk IDs are within the expected range.
    Based on our exploration, chunks go from 0000 to 3115.
    """
    valid_chunks = True
    for chunk_id in chunk_ids:
        try:
            chunk_num = int(chunk_id)
            if not 0 <= chunk_num <= 3115:
                print(
                    f"Warning: Chunk {chunk_id} may be outside expected range (0000-3115)"
                )
        except ValueError:
            print(f"Error: Invalid chunk format '{chunk_id}'. Must be a number.")
            valid_chunks = False
    return valid_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Download PhysicalAI-Autonomous-Vehicles dataset front wide camera, ego motion, and calibration data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Download first chunk (0000)
  %(prog)s --chunk 0001              # Download second chunk
  %(prog)s --chunk 0000 0001 0002    # Download multiple chunks
  %(prog)s --all                    # Download all available chunks (WARNING: ~6TB)
        """,
    )

    parser.add_argument(
        "--chunk",
        nargs="+",
        help="Chunk IDs to download (e.g., 0000, 0001, etc.). Default: 0000",
        default=["0000"],
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all chunks (WARNING: this will be extremely large, ~6TB)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="physical_ai_data",
        help="Output directory for downloaded files (default: physical_ai_data)",
    )

    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        print("Preparing to download all chunks...")
        print("WARNING: This will download ~6TB of data!")
        response = input("Are you sure you want to continue? (y/N): ")
        if response.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

        # Generate all chunk IDs from 0000 to 3115
        args.chunk = [f"{i:04d}" for i in range(3116)]
        print(f"Will download {len(args.chunk)} chunks.")

    # Validate chunk format
    for chunk_id in args.chunk:
        if not chunk_id.isdigit() or len(chunk_id) != 4:
            print(f"Error: Chunk ID '{chunk_id}' must be a 4-digit number (e.g., 0000)")
            sys.exit(1)

    # Validate chunk range
    if not validate_chunk_range(args.chunk):
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PhysicalAI-Autonomous-Vehicles Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")

    # Check authentication
    if not authenticate():
        print(
            "Authentication failed. Please ensure you have access to the gated dataset."
        )
        sys.exit(1)

    # Download data
    repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"

    print(f"Downloading chunks: {', '.join(args.chunk)}")
    print(
        "Target data: Front wide camera + Ego motion + Camera intrinsics + Sensor extrinsics + Vehicle dimensions"
    )

    # Show estimated size warning for large downloads
    if len(args.chunk) > 10:
        estimated_size_gb = (
            len(args.chunk) * 2.1
        )  # Rough estimate: ~2.1GB per chunk (including calibration)
        print(
            f"\nWARNING: This download may be approximately {estimated_size_gb:.0f}GB"
        )
        print(
            "Make sure you have sufficient disk space and stable internet connection."
        )
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

    # Get files to download
    files_to_download = get_chunk_files(args.chunk)
    total_files = len(files_to_download)
    print(f"Attempting to download {total_files} files...")

    # If force flag is set, remove existing files
    if args.force:
        print("\nForce mode: Removing existing files...")
        removed_count = 0
        for _, local_name in files_to_download:
            if local_name.endswith(".zip"):
                chunk_id = local_name.split("_chunk_")[1].replace(".zip", "")
                if "camera_front_wide" in local_name:
                    extract_dir = output_dir / f"camera_chunk_{chunk_id}"
                elif "ego_motion" in local_name:
                    extract_dir = output_dir / f"ego_motion_chunk_{chunk_id}"
                else:
                    extract_dir = output_dir / f"chunk_{chunk_id}"
                if extract_dir.exists():
                    import shutil

                    shutil.rmtree(extract_dir)
                    removed_count += 1
            else:
                local_path = output_dir / local_name
                if local_path.exists():
                    local_path.unlink()
                    removed_count += 1
        if removed_count > 0:
            print(f"Removed {removed_count} existing files/directories.")

    # Download files
    downloaded_files = download_files(
        repo_id=repo_id,
        files=files_to_download,
        output_dir=output_dir,
        show_progress=not args.no_progress,
    )

    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"Successfully downloaded: {len(downloaded_files)}/{total_files} items")
    print(f"Output directory: {output_dir.absolute()}")

    if downloaded_files:
        total_size_mb = sum(
            f.stat().st_size for f in downloaded_files if f.is_file()
        ) / (1024**2)
        total_size_gb = total_size_mb / 1024
        print(f"Total downloaded size: {total_size_gb:.2f}GB")

        print("\nDownloaded/Extracted files:")
        for file_path in downloaded_files:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024**2)
                print(f"  {file_path.name} ({size_mb:.1f}MB)")
            else:
                # Count files in directory
                file_count = len([f for f in file_path.rglob("*") if f.is_file()])
                print(f"  {file_path.name}/ ({file_count} files extracted)")

        # Organize files by type for better readability
        camera_dirs = [
            f
            for f in downloaded_files
            if f.is_dir() and f.name.startswith("camera_chunk_")
        ]
        ego_motion_dirs = [
            f
            for f in downloaded_files
            if f.is_dir() and f.name.startswith("ego_motion_chunk_")
        ]
        intrinsics_files = [
            f for f in downloaded_files if f.is_file() and "camera_intrinsics" in f.name
        ]
        extrinsics_files = [
            f for f in downloaded_files if f.is_file() and "sensor_extrinsics" in f.name
        ]
        vehicle_files = [
            f
            for f in downloaded_files
            if f.is_file() and "vehicle_dimensions" in f.name
        ]

        print(f"\nData summary:")
        print(f"  Camera directories: {len(camera_dirs)} (videos + metadata)")
        print(f"  Ego motion directories: {len(ego_motion_dirs)} (motion data)")
        print(f"  Camera intrinsics: {len(intrinsics_files)}")
        print(f"  Sensor extrinsics: {len(extrinsics_files)}")
        print(f"  Vehicle dimensions: {len(vehicle_files)}")

    print("=" * 60)

    if len(downloaded_files) == 0:
        print("No files were downloaded successfully.")
        print("This could mean:")
        print("- The chunk(s) don't exist")
        print("- You don't have access to the dataset")
        print("- Network connectivity issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
