msggen:

0:
export VERTEXAI_LOCATION=us-west1; export VERTEXAI_PROJECT=triangulate-396717; export GOOGLE_CLOUD_PROJECT=triangulate-396717
# uv run mini-extra swebench -c configs/my_sbv_gemini.yaml --subset syn_django_20201231_e13b714 --split test --workers 24 --output evals/syn/django_20201231_e13b714

LOG_LEVEL=WARNING script -c 'uv run python -u src/repotune/datasyn/gen_issues.py --input_repo astropy/astropy --output_dir data/syn/bugfix/astropy_20201231_72ca3c07 --commit_hash 72ca3c07 --model vertex_ai/gemini-2.5-pro --temperature 1.0 --reasoning_effort medium --num_workers 16 --num_samples -1 --use_git_history False --image_name jyangballin/swesmith.x86_64.astropy_1776_astropy.72ca3c07:latest run' data/syn/bugfix/astropy_20201231_72ca3c07/gen_v0.log

1:
export VERTEXAI_LOCATION=europe-west4; export VERTEXAI_PROJECT=triangulate-396717; export GOOGLE_CLOUD_PROJECT=triangulate-396717
uv run mini-extra swebench -c configs/my_sbv_gemini.yaml --subset syn_sympy_20201231_6f92459 --split test --workers 24 --output evals/syn/sympy_20201231_6f92459



2:
export VERTEXAI_LOCATION=us-east4; export VERTEXAI_PROJECT=triangulate-396717; export GOOGLE_CLOUD_PROJECT=triangulate-396717
# uv run mini-extra swebench -c configs/my_sbv_gemini.yaml --subset syn_matplotlib_20201231_5b89c9c5 --split test --workers 24 --output evals/syn/matplotlib_20201231_5b89c9c5

LOG_LEVEL=WARNING script -c 'uv run python -u src/repotune/datasyn/gen_issues.py --input_repo pydata/xarray --output_dir data/syn/bugfix/xarray_20201231_81c8ac99 --commit_hash 81c8ac99 --model vertex_ai/gemini-2.5-pro --temperature 1.0 --reasoning_effort medium --num_workers 16 --num_samples -1 --use_git_history False --image_name jyangballin/swesmith.x86_64.pydata_1776_xarray.81c8ac99:latest run' data/syn/bugfix/xarray_20201231_81c8ac99/gen_v0.log

3:
export VERTEXAI_LOCATION=europe-west1; export VERTEXAI_PROJECT=triangulate-396717; export GOOGLE_CLOUD_PROJECT=triangulate-396717
# uv run mini-extra swebench -c configs/my_sbv_gemini.yaml --subset syn_sphinx_20201231_4b45233 --split test --workers 24 --output evals/syn/sphinx_20201231_4b4523

LOG_LEVEL=WARNING script -c 'uv run python -u src/repotune/datasyn/gen_issues.py --input_repo scikit-learn/scikit-learn --output_dir data/syn/bugfix/scikit-learn_20201222_6b4f824 --commit_hash 6b4f824 --model vertex_ai/gemini-2.5-pro --temperature 1.0 --reasoning_effort medium --num_workers 16 --num_samples -1 --use_git_history False --image_name jyangballin/swesmith.x86_64.scikit-learn_1776_scikit-learn.6b4f8243:latest run' data/syn/bugfix/scikit-learn_20201222_6b4f824/gen_v0.log

ovh-0:
export VERTEXAI_LOCATION=europe-west8; export VERTEXAI_PROJECT=triangulate-396717; export GOOGLE_CLOUD_PROJECT=triangulate-396717
uv run mini-extra swebench -c configs/my_sbv_gemini.yaml --subset swesmith --split test --workers 24 --output evals/syn/swesmith

ovh-1:
# export VERTEXAI_LOCATION=europe-west9; export VERTEXAI_PROJECT=triangulate-396717; export GOOGLE_CLOUD_PROJECT=triangulate-396717
# uv run mini-extra swebench -c configs/my_sbv_gemini.yaml --subset swesmith --split test --workers 24 --output evals/syn/swesmith



vllm serve /data/repotune/train/outputs/classic/django_classic_all_r32_qwen3coder30i_16k_2ep_cos_1215/v0-20251216-043134/checkpoint-48 --served-model-name django_classic_all_r32_qwen3coder30i_16k_2ep_cos_1215_ckpt48 --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser qwen3_coder --api-key colin --port 8001

uv run mini-extra swebench -c my_sbv.yaml --subset verified --split test --workers 16 --output evals/django_e13b714/sbv_full_qwen3coder30i

uv run sb-cli submit swe-bench_verified test --predictions_path evals/.../preds.json --run_id ...

