vllm serve /data/repotune/train/outputs/classic/django_classic_all_r32_qwen3coder30i_16k_2ep_cos_1215/v0-20251216-043134/checkpoint-48 --served-model-name django_classic_all_r32_qwen3coder30i_16k_2ep_cos_1215_ckpt48 --enable-expert-parallel --enable-auto-tool-choice --tool-call-parser qwen3_coder --api-key colin --port 8001

uv run mini-extra swebench -c my_sbv.yaml --subset verified --split test --workers 16 --output evals/django_e13b714/sbv_full_qwen3coder30i

uv run sb-cli submit swe-bench_verified test --predictions_path evals/.../preds.json --run_id ...

