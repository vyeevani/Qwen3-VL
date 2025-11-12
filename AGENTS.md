# Repository Guidelines

## Project Structure & Module Organization
Fine-tuning code (datasets, trainers, DeepSpeed presets) is located in `qwen-vl-finetune/qwenvl`, while exploratory notebooks and helper scripts live in `cookbooks/` alongside their referenced media in `cookbooks/assets`. Production-ready preprocessing utilities ship from `qwen-vl-utils/src/qwen_vl_utils`, evaluation flows (such as MMMU) sit in `evaluation/mmmu`, and the lightweight multimodal demo runs through `web_demo_mm.py` or `docker/`.

## Build, Test, and Development Commands
Install and launch the browser demo when tweaking UX:
```bash
pip install -r requirements_web_demo.txt
python web_demo_mm.py -c /path/to/qwen3vl/weights
cd docker && bash run_web_demo.sh -c /path/to/qwen3vl/weights --port 8881
```
Model serving follows `vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --rope-scaling ... --max-model-len 1000000`; reuse that template when touching deployment notes. Reuse the torchrun block in `qwen-vl-finetune/README.md` for training, and pair every dataset change with `python qwen-vl-finetune/tools/pack_data.py ...` or `check_image.py`. For evaluations, install `evaluation/mmmu/requirements.txt` and run the documented `infer` then `eval` commands.

## Coding Style & Naming Conventions
Python dominates the repo; adhere to Ruff’s config (`line-length = 119`, double quotes, spaces) defined in `qwen-vl-utils/pyproject.toml`. Use snake_case for modules/functions, PascalCase for classes, and reserve CLI entry points for top-level scripts. Prefer typed signatures, concise docstrings, and keep notebook cells reproducible (self-contained setup + execute). Run `ruff check .` and `ruff format .` before committing Python edits.

## Testing Guidelines
Demonstrate behavioral impact instead of adding ad-hoc prints. Code-generation flows should pass `python cookbooks/utils/multimodal_coding/test_mmcode.py`. For benchmarks, execute `python evaluation/mmmu/run_mmmu.py infer ...` followed by `eval ...` and attach the generated accuracy JSON/CSV. Data contributions should include the output of `tools/check_image.py` (missing assets) and any packer logs; large training changes ought to cite loss curves or short validation runs.

## Commit & Pull Request Guidelines
Recent history favors concise, imperative commits with optional tags like `[FIX]` or `[DOC]`; keep each commit scoped to one logical change and avoid noisy WIP stacks. PRs must summarize intent, list commands/tests executed, reference related issues or cookbooks, and include artifacts (screenshots, metrics, log excerpts) whenever behavior changes. Highlight required checkpoints, API keys, or GPU constraints so reviewers can reproduce results quickly.

## Security & Configuration Notes
Never check in credentials. MMMU evaluation needs `CHATGPT_DASHSCOPE_API_KEY`, `DASHSCOPE_API_BASE`, or `MIT_SPIDER_TOKEN`/`MIT_SPIDER_URL`; load them via environment variables or local secrets files ignored by git. Store public sample data in `cookbooks/assets`, keep proprietary paths in local configs, and document CUDA/driver expectations whenever you update Dockerfiles or requirements.
