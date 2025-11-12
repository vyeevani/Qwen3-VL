# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-VL is a state-of-the-art vision-language model supporting image and video understanding. The repository contains:
- **Model inference code**: For running Qwen3-VL models (Instruct and Thinking variants) ranging from 2B to 235B parameters
- **Fine-tuning framework**: Located in `qwen-vl-finetune/` for training on custom vision-language data
- **Vision preprocessing utilities**: The `qwen-vl-utils` package for handling image/video inputs
- **Cookbooks**: Jupyter notebooks demonstrating specific capabilities (OCR, grounding, agents, etc.)
- **Evaluation tools**: MMMU benchmark evaluation in `evaluation/mmmu/`

## Common Commands

### Environment Setup

**Initialize the repository environment with uv:**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (basic installation)
uv sync

# Sync with download dependencies (for dataset downloading)
uv sync --extra download

# Sync with all optional dependencies
uv sync --all-extras

# Run scripts using uv
uv run python script.py
```

### Data Preparation Pipeline

**Step 1: Download PhysicalAI-Autonomous-Vehicles dataset:**
```bash
# Download first chunk (default)
uv run python download_dataset.py

# Download specific chunk(s)
uv run python download_dataset.py --chunk 0001
uv run python download_dataset.py --chunk 0000 0001 0002

# Download all chunks (WARNING: ~6TB)
uv run python download_dataset.py --all

# Specify output directory
uv run python download_dataset.py --chunk 0000 --output-dir /path/to/data

# Force re-download
uv run python download_dataset.py --chunk 0000 --force

# Requires Hugging Face authentication:
uv run huggingface-cli login
# Or set HF_TOKEN environment variable
```

**Step 2: Prepare video segments for training:**
```bash
# Install video processing dependencies
uv sync --extra video

# Extract 1-second video segments with motion targets (test on a few clips)
uv run python prepare_video_segments.py --max_clips 5

# Process all clips in chunk
uv run python prepare_video_segments.py

# Custom input/output directories
uv run python prepare_video_segments.py \
  --input_dir physical_ai_data \
  --output_dir qwen_video_segments \
  --fps 30

# Output structure:
# qwen_video_segments/
#   video_segments/      # 1-second video clips (.mp4)
#   metadata/
#     segments.json      # Segment metadata with motion targets
```

The preparation script:
- Extracts 19 overlapping 1-second segments from each 20-second clip
- Calculates velocity and curvature targets for the next second
- Creates training pairs: (1-second video) → (velocity, curvature)
- Outputs formatted motion targets ready for Qwen-VL training

**Step 3: Create Hugging Face dataset:**
```bash
# Convert to HF dataset format (test on subset first)
uv run python prepare_huggingface_dataset.py --max-samples 10 --overwrite-output

# Process all segments
uv run python prepare_huggingface_dataset.py --overwrite-output

# Custom configuration
uv run python prepare_huggingface_dataset.py \
  --segments-path qwen_video_segments/metadata/segments.json \
  --ego-motion-dir physical_ai_data/ego_motion_chunk_0000 \
  --output-dir qwen_video_dataset \
  --test-ratio 0.1 \
  --fps 30 \
  --overwrite-output

# Output structure:
# qwen_video_dataset/
#   train/              # Training split (HF dataset)
#   test/               # Test split (HF dataset)
#   train_examples.json # Preview of 5 training examples
#   test_examples.json  # Preview of 5 test examples
```

The HF dataset preparation script:
- Resamples ego motion to 30 velocity/curvature pairs per second
- Creates proper Qwen-VL conversation format (system + user + assistant)
- Splits data into train/test (default 90/10)

**Step 4: Fine-tune the model (see TRAINING.md for details):**
```bash
# Install fine-tuning dependencies
uv sync --extra finetune

# Convert dataset to training format
./convert_to_training_format.py

# Option A: LoRA training (recommended, lower memory)
./train_physical_ai_lora.sh

# Option B: Full fine-tuning (higher memory)
./train_physical_ai.sh

# See TRAINING.md for detailed configuration, troubleshooting, and inference
```

### Running Inference

**Using Transformers:**
```bash
# Install dependencies (requires transformers >= 4.57.0)
pip install "transformers>=4.57.0"
pip install qwen-vl-utils==0.0.14

# For image/video inference, see examples in README.md
```

**Using vLLM (recommended for production):**
```bash
# Install
pip install accelerate qwen-vl-utils==0.0.14
uv pip install -U vllm  # vllm >= 0.11.0

# Start server (example for H100/H200)
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --host 0.0.0.0 \
  --port 22002
```

**Using SGLang (alternative):**
```bash
python -m sglang.launch_server \
   --model-path Qwen/Qwen3-VL-235B-A22B-Instruct \
   --host 0.0.0.0 \
   --port 22002 \
   --tp 4
```

### Web Demo

```bash
# Install dependencies
pip install -r requirements_web_demo.txt

# Launch web UI
python web_demo_mm.py -c /path/to/qwen3vl/weights

# Or use Docker
cd docker && bash run_web_demo.sh -c /path/to/qwen3vl/weights --port 8881
```

### Fine-tuning

**Training setup:**
```bash
# Single test: Run the demo dataset first
cd qwen-vl-finetune
torchrun --nproc_per_node=<NUM_GPUS> \
         qwenvl/train/train_qwen.py \
         --model_name_or_path /path/to/Qwen3-VL-model \
         --dataset_use your_dataset%100 \
         --output_dir ./checkpoints \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --num_train_epochs 3 \
         --deepspeed zero3.json

# See scripts/sft_32b.sh for 32B model example
```

**Data preparation:**
```bash
# Check for missing images in dataset
python qwen-vl-finetune/tools/check_image.py --dataset_path /path/to/annotations.json

# Pack data for efficient training
python qwen-vl-finetune/tools/pack_data.py --input /path/to/data.json --output /path/to/packed_data.json
```

### Development (qwen-vl-utils)

```bash
cd qwen-vl-utils

# Code formatting and linting (Ruff config in pyproject.toml)
ruff check .
ruff format .

# The library is published to PyPI - build with hatchling
```

### Evaluation

```bash
cd evaluation/mmmu

# Install requirements
pip install -r requirements.txt

# Set environment variables for API access
export CHATGPT_DASHSCOPE_API_KEY="your_key"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

# Run evaluation (example)
python run_mmmu.py infer --model_path /path/to/model --output results/
python run_mmmu.py eval --results_dir results/
```

## Architecture & Code Organization

### Directory Structure

- **`qwen-vl-finetune/qwenvl/`**: Core training framework
  - `train/`: Training loop (`train_qwen.py`), custom `Trainer`, arguments
  - `data/`: Dataset configurations (`__init__.py` registers datasets), data processors, RoPE implementations

- **`qwen-vl-utils/src/qwen_vl_utils/`**: Production vision preprocessing
  - Handles image/video loading from URLs, local paths, base64
  - Supports multiple video backends: torchvision, decord, torchcodec
  - Process vision info with `process_vision_info()` function

- **`cookbooks/`**: Capability demonstrations via Jupyter notebooks
  - Each notebook is self-contained with setup cells
  - `cookbooks/assets/`: Referenced media files
  - `cookbooks/utils/`: Shared utility scripts

- **`evaluation/mmmu/`**: MMMU benchmark evaluation pipeline

- **`docker/`**: Dockerfile and scripts for containerized deployment

### Model Architecture Notes

**Qwen3-VL introduces:**
1. **Interleaved-MRoPE**: Position embeddings for time, width, and height dimensions in video
2. **DeepStack**: Multi-level ViT feature fusion for fine-grained image-text alignment
3. **Text-Timestamp Alignment**: Precise temporal event localization in videos

**Model variants:**
- **Instruct**: Standard instruction-following models
- **Thinking**: Enhanced reasoning with extended context (use higher `out_seq_length`)
- **MoE (e.g., 235B-A22B)**: Mixture-of-Experts architecture; does NOT support DeepSpeed ZeRO-3

### Training Data Format

Training expects JSONL with this structure:
```json
{
  "image": "path/to/image.jpg",  // or ["img1.jpg", "img2.jpg"] for multi-image
  "video": "path/to/video.mp4",  // mutually exclusive with "image"
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this image."},
    {"from": "gpt", "value": "A cat sitting on a table."}
  ]
}
```

**Important constraints:**
- Each `<image>` tag must correspond to exactly one image in the `image` field
- Each `<video>` tag must correspond to exactly one video in the `video` field
- Tags should NOT appear in assistant responses
- Register datasets in `qwen-vl-finetune/qwenvl/data/__init__.py` with sampling rates (e.g., `"dataset%50"` = 50% sampling)

### Key Training Parameters

**Component tuning flags:**
- `tune_mm_llm`: Train the language model backbone
- `tune_mm_vision`: Train the vision encoder (should be False when training with both image and video)
- `tune_mm_mlp`: Train the multimodal projector

**Resolution control:**
- Images: `--max_pixels 576*28*28 --min_pixels 16*28*28` (for Qwen2.5-VL; adjust to 32x32 patches for Qwen3-VL)
- Videos: `--video_max_pixels 1664*28*28 --video_min_pixels 256*28*28`
- Video sampling: `--video_fps 2 --video_max_frames 8`

**Data efficiency:**
- `--data_flatten True`: Concatenate batch sequences into one sequence
- `--data_packing True`: Use pre-packed data (requires `tools/pack_data.py`)

**LoRA fine-tuning:**
```bash
--lora_enable True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.0
```

### Context Length Extension

Qwen3-VL supports native 256K context, extendable to 1M via YaRN:
- Modify `max_position_embeddings` and `rope_scaling` in `config.json`
- Use smaller scaling factors (2-3) due to Interleaved-MRoPE's slower position ID growth
- For vLLM: Add `--rope-scaling '{"rope_type":"yarn",...}' --max-model-len 1000000`

### Vision Processing Details

**Pixel budget control (Qwen3-VL):**
- Compression ratio: 32 (so 256 tokens = 256×32×32 pixels)
- Image processor: `size['longest_edge']` = max pixels, `size['shortest_edge']` = min pixels
- Video processor: `size['longest_edge']` = total pixels across all frames (T×H×W)

**Using qwen-vl-utils:**
- Set `image_patch_size=16` for Qwen3-VL (14 for Qwen2.5-VL)
- Set `return_video_metadata=True` for Qwen3-VL to get video metadata
- Pass `do_resize=False` to processor when using qwen-vl-utils (it already resizes)

## Code Style

**Python:**
- Line length: 119 characters (configured in `qwen-vl-utils/pyproject.toml`)
- Quote style: Double quotes
- Indentation: Spaces
- Naming: snake_case for functions/modules, PascalCase for classes
- Use type hints and concise docstrings
- Run `ruff check .` and `ruff format .` before committing

**Jupyter Notebooks:**
- Keep cells reproducible and self-contained
- Include setup cells at the top

**Commit messages:**
- Use imperative mood, concise format
- Optional tags: `[FIX]`, `[DOC]`, `[FEAT]` etc.
- One logical change per commit

## Important Notes

### Security
- Never commit API keys or credentials
- Load secrets from environment variables: `CHATGPT_DASHSCOPE_API_KEY`, `DASHSCOPE_API_BASE`, `MIT_SPIDER_TOKEN`, etc.
- Store public sample data in `cookbooks/assets/`

### GPU Requirements
- 32B model training requires 8×80GB GPUs (see `scripts/sft_32b.sh`)
- MoE models do not support DeepSpeed ZeRO-3
- Enable Flash Attention 2 by adding `"_attn_implementation": "flash_attention_2"` in model config.json

### Video Backend Selection
- Default: torchvision (slowest, but widely compatible)
- Faster: decord (Linux PyPI available) or torchcodec (requires FFmpeg)
- Force backend: `export FORCE_QWENVL_VIDEO_READER=torchcodec`
- URL support varies: torchvision ≥0.19.0 and torchcodec support HTTPS; decord only HTTP

### Generation Hyperparameters

**Instruct models:**
```bash
top_p=0.8, top_k=20, temperature=0.7
repetition_penalty=1.0, presence_penalty=1.5
max_tokens=32768
```

**Thinking models:**
```bash
top_p=0.95, top_k=20, temperature=0.6
repetition_penalty=1.0, presence_penalty=0.0
max_tokens=40960
```

## Reference Files

- Main README: `/root/Qwen3-VL/README.md` - Comprehensive usage guide and model details
- Training guide: `/root/Qwen3-VL/qwen-vl-finetune/README.md` - Fine-tuning instructions
- Repository guidelines: `/root/Qwen3-VL/AGENTS.md` - Detailed contribution and development guidelines
- Utils README: `/root/Qwen3-VL/qwen-vl-utils/README.md` - Vision preprocessing library docs
