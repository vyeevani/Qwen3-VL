# Fine-tuning Qwen3-VL on Physical AI Driving Dataset

This guide explains how to fine-tune Qwen3-VL models to predict ego motion (velocity and curvature) from 1-second driving video clips.

## Prerequisites

### 1. Hardware Requirements

**Minimum (LoRA training):**
- 1x GPU with 24GB+ VRAM (e.g., RTX 3090, RTX 4090, A5000)
- 32GB system RAM
- 50GB free disk space

**Recommended (full fine-tuning):**
- 2x GPUs with 40GB+ VRAM (e.g., A100, H100)
- 64GB system RAM
- 100GB free disk space

### 2. Software Requirements

Install fine-tuning dependencies:

```bash
# Install all fine-tuning dependencies
uv sync --extra finetune

# This includes:
# - deepspeed>=0.17.0
# - peft>=0.17.0
# - flash-attn>=2.7.0
# - triton>=3.2.0
```

**Note**: `flash-attn` requires CUDA and may take several minutes to build from source.

## Pipeline Overview

### Step 1: Prepare Training Data (Already Done)

If you followed the main pipeline, you already have:
- ✅ Raw video clips downloaded (`physical_ai_data/`)
- ✅ 1-second segments extracted (`qwen_video_segments/`)
- ✅ HF dataset created (`qwen_video_dataset/`)

### Step 2: Convert Dataset Format

The Qwen-VL training framework expects a different format than the HF dataset. Convert it:

```bash
./convert_to_training_format.py

# Output:
# - qwen_video_dataset_training/train.json
# - qwen_video_dataset_training/test.json
```

**Format difference:**
```json
// Before (HF format)
{
  "messages": [
    {"role": "user", "content": [{"type": "video", "video": "path.mp4"}, ...]}
  ]
}

// After (Training format)
{
  "video": "path.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nQuestion here"},
    {"from": "gpt", "value": "Answer here"}
  ]
}
```

### Step 3: Choose Training Approach

Two options:

#### Option A: LoRA Fine-tuning (Recommended)

**Pros:**
- Lower memory requirements (can train on 1x 24GB GPU)
- Faster training
- Smaller output (only adapter weights, ~100MB)
- Less risk of catastrophic forgetting

**Cons:**
- Slightly lower performance ceiling
- Requires merging adapters for deployment

**Command:**
```bash
./train_physical_ai_lora.sh
```

#### Option B: Full Fine-tuning

**Pros:**
- Maximum performance
- No merging needed
- Full model saved directly

**Cons:**
- High memory requirements (requires 2x 40GB GPUs with DeepSpeed)
- Slower training
- Larger output (full model, ~8GB for 4B variant)

**Command:**
```bash
./train_physical_ai.sh
```

## Training Configuration

### Dataset Configuration

Registered in `qwen-vl-finetune/qwenvl/data/__init__.py`:

```python
PHYSICAL_AI_DRIVING = {
    "annotation_path": "qwen_video_dataset_training/train.json",
    "data_path": "",  # Paths in annotation are relative to project root
}
```

### Video Processing Parameters

Our dataset has specific characteristics:
- **FPS**: 30 (1 second = 30 frames exactly)
- **Duration**: 1.0 seconds per clip
- **Output**: 30 velocity/curvature pairs

Training parameters are set accordingly:

```bash
VIDEO_FPS=30              # Match source video FPS
VIDEO_MAX_FRAMES=32       # Slightly higher to accommodate full clip
VIDEO_MIN_FRAMES=16       # Lower bound for adaptive sampling
VIDEO_MAX_PIXELS=1024×28×28  # ~805K pixels (adjust for memory)
VIDEO_MIN_PIXELS=256×28×28   # ~200K pixels
```

### Training Hyperparameters

**Full fine-tuning:**
- Learning rate: `1e-5`
- Batch size: `2` per GPU
- Gradient accumulation: `4` steps
- Epochs: `3`
- Optimizer: AdamW
- Schedule: Cosine with 3% warmup

**LoRA fine-tuning:**
- Learning rate: `2e-4` (higher for LoRA)
- Batch size: `4` per GPU
- Gradient accumulation: `2` steps
- Epochs: `5` (more epochs for LoRA)
- LoRA r: `64`
- LoRA alpha: `128`
- LoRA dropout: `0.05`

### Components Being Trained

```bash
--tune_mm_vision False  # Vision encoder frozen (recommended for stability)
--tune_mm_mlp True      # Train vision-language projection
--tune_mm_llm True      # Train language model
```

For even more efficient training, you can freeze the LLM:
```bash
--tune_mm_llm False  # Only train the projection layer
```

## Running Training

### 1. Full Dataset Training

Process all segments first:

```bash
# Generate all 1-second segments from all 100 clips
uv run python prepare_video_segments.py

# Create full HF dataset
uv run python prepare_huggingface_dataset.py --overwrite-output

# Convert to training format
./convert_to_training_format.py

# Launch LoRA training
./train_physical_ai_lora.sh
```

Expected dataset size: ~1,900 training examples (19 segments × 100 clips)

### 2. Subset Training (Testing)

Already done with our 10-sample test dataset:

```bash
# Training data already exists at:
# - qwen_video_dataset_training/train.json (5 samples)
# - qwen_video_dataset_training/test.json (1 sample)

# Quick test (will run but likely underfit)
EPOCHS=1 BATCH_SIZE=2 ./train_physical_ai_lora.sh
```

### 3. Multi-GPU Training

Automatically detected, or manually specify:

```bash
# Use 2 GPUs
NPROC_PER_NODE=2 ./train_physical_ai_lora.sh

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 ./train_physical_ai_lora.sh
```

### 4. Resume from Checkpoint

If training is interrupted:

```bash
# Training automatically resumes from latest checkpoint if found
./train_physical_ai_lora.sh
```

The script checks `OUTPUT_DIR` for existing `checkpoint-*` directories.

## Monitoring Training

### Progress Logs

Training outputs real-time metrics:

```
{'loss': 2.456, 'learning_rate': 1.5e-05, 'epoch': 0.1}
{'loss': 2.123, 'learning_rate': 1.8e-05, 'epoch': 0.2}
...
```

### Enable Weights & Biases

```bash
# Install wandb
uv pip install wandb

# Login
wandb login

# Modify training script
# Change: --report_to "none"
# To:     --report_to "wandb" --run_name "physical_ai_v1"
```

### Enable TensorBoard

```bash
# Modify training script
# Change: --report_to "none"
# To:     --report_to "tensorboard"

# View logs
tensorboard --logdir checkpoints/physical_ai_driving_lora/
```

## After Training

### LoRA: Merge Adapters (Optional)

To create a standalone model:

```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Load base model
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "checkpoints/physical_ai_driving_lora")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("physical_ai_merged")

# Save processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
processor.save_pretrained("physical_ai_merged")
```

### Full Fine-tuning: Use Directly

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "checkpoints/physical_ai_driving",
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("checkpoints/physical_ai_driving")
```

## Inference with Fine-tuned Model

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load fine-tuned model (LoRA or full)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "checkpoints/physical_ai_driving_lora",  # or merged model
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# Prepare input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Given this 1-second, 30 FPS driving clip, output exactly 30 pairs of (velocity in m/s, curvature) for the next second as [(xx.x, 0.xxx), ...]. No prose, just the list."},
            {"type": "video", "video": "path/to/test_video.mp4", "fps": 30}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

# Generate prediction
output_ids = model.generate(**inputs, max_new_tokens=512)
output_text = processor.batch_decode(
    output_ids[inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(output_text)
# Expected: [(xx.x, 0.xxx), (xx.x, 0.xxx), ...]
```

## Troubleshooting

### Out of Memory

**Solutions:**
1. Reduce batch size: `BATCH_SIZE=1 ./train_physical_ai_lora.sh`
2. Reduce video resolution:
   ```bash
   VIDEO_MAX_PIXELS=$((512 * 28 * 28)) ./train_physical_ai_lora.sh
   ```
3. Reduce max frames:
   ```bash
   VIDEO_MAX_FRAMES=16 ./train_physical_ai_lora.sh
   ```
4. Enable CPU offload (edit DeepSpeed config):
   ```bash
   DEEPSPEED_CONFIG="qwen-vl-finetune/scripts/zero3_offload.json"
   ```

### Flash Attention Installation Fails

If `flash-attn` fails to install:

```bash
# Train without flash attention
# Edit train script, change line:
# train(attn_implementation="flash_attention_2")
# to:
# train(attn_implementation="eager")
```

Or manually install prerequisites:

```bash
# Ensure CUDA is available
nvidia-smi

# Install ninja for faster builds
uv pip install ninja

# Retry flash-attn install
uv pip install flash-attn --no-build-isolation
```

### Model Loading Errors

If you see errors about model class:

```python
# The training script auto-detects model type from path
# Ensure model path contains "qwen3" for Qwen3-VL models
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"  # ✓ Correct
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"  # ✗ Wrong (Qwen2)
```

### Dataset Not Found

```bash
# Ensure dataset config points to correct path
ls qwen_video_dataset_training/train.json

# If missing, regenerate:
./convert_to_training_format.py
```

## Expected Training Time

**On 1x RTX 4090 (24GB), LoRA training:**
- 10 samples: ~5 minutes
- 100 samples: ~30 minutes
- 1,900 samples: ~10 hours

**On 2x A100 (80GB), full fine-tuning:**
- 1,900 samples: ~6 hours

Times are approximate and depend on:
- Video resolution settings
- Number of frames per video
- Batch size and gradient accumulation
- Model size (4B vs 7B vs 32B)

## Next Steps

After successful training:

1. **Evaluate on test set**: Run inference on held-out videos
2. **Visualize predictions**: Plot predicted vs ground truth trajectories
3. **Calculate metrics**: MSE, MAE for velocity and curvature
4. **Iterate**: Adjust hyperparameters based on results
5. **Scale up**: Train on more chunks of the PhysicalAI dataset

## References

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [PEFT Library (LoRA)](https://github.com/huggingface/peft)
- [DeepSpeed](https://www.deepspeed.ai/)
- [PhysicalAI Dataset](https://huggingface.co/datasets/PhysicalAI/autonomous-vehicles)
