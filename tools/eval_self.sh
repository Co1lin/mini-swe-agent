set -x

PORT=${PORT:-0}
MODEL=${MODEL:-openai/Qwen/Qwen3-4B-Instruct-2507}
MS=${MS:-$MODEL}
REPO=${REPO:-django}
HASH=${HASH:-e13b714}
VERSION=${VERSION:-1}
CONFIG=${CONFIG:-configs/my_sbv.yaml}
WORKERS=${WORKERS:-12}

REPO_HASH=${REPO}_${HASH}
OUTPUT_DIR=evals/$REPO_HASH/$MS/v$VERSION
RUN_ID=bugfix_${REPO_HASH}_${MS}_v${VERSION}

uv run mini-extra swebench -c $CONFIG --subset verified --split test --workers $WORKERS \
    --instance-ids-paths /home/colin/code/repotune/data/val/$REPO_HASH/sbv_${REPO_HASH}_ids.json \
    --model $MODEL \
    --remote-port-selection $PORT \
    --output $OUTPUT_DIR

sleep 3s

uv run python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path $OUTPUT_DIR/preds.json \
    --max_workers 16 \
    --run_id $RUN_ID

mv *$RUN_ID.json $OUTPUT_DIR/

uv run tools/eval_loc.py --pred_file $OUTPUT_DIR/preds.json

uv run tools/stat.py run --eval_dir $OUTPUT_DIR

grep _instances $OUTPUT_DIR/*$RUN_ID.json
