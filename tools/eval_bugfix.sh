set -x

PORT=${PORT:-0}
MS=${MS:-unknown}
MODEL=${MODEL:-openai/$MS}
MODEL_CLASS=${MODEL_CLASS:-litellm}
REPO=${REPO:-django}
HASH=${HASH:-e13b714}
VERSION=${VERSION:-1}
CONFIG=${CONFIG:-configs/my_sbv.yaml}
WORKERS=${WORKERS:-12}
RUN_EVAL=${RUN_EVAL:-true}

REPO_HASH=${REPO}_${HASH}
OUTPUT_DIR=evals/$REPO_HASH/$MS/v$VERSION
RUN_ID=bugfix_${REPO_HASH}_${MS}_v${VERSION}

uv run mini-extra swebench -c $CONFIG --subset verified --split test --workers $WORKERS \
    --instance-ids-paths /home/colin/code/repotune/data/val/$REPO_HASH/sbv_${REPO_HASH}_ids.json \
    --model $MODEL \
    --model-class $MODEL_CLASS \
    --remote-port-selection $PORT \
    --output $OUTPUT_DIR

if [ "$RUN_EVAL" = "true" ]; then
    uv run python -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_Verified \
        --predictions_path $OUTPUT_DIR/preds.json \
        --max_workers 12 \
        --run_id $RUN_ID

    mv *$RUN_ID.json $OUTPUT_DIR/

    uv run tools/eval_loc.py --pred_file $OUTPUT_DIR/preds.json 2>&1 | tee -a $OUTPUT_DIR/report.log

    uv run tools/stat.py run --eval_dir $OUTPUT_DIR | tee -a $OUTPUT_DIR/report.log

    grep _instances $OUTPUT_DIR/*$RUN_ID.json | tee -a $OUTPUT_DIR/report.log
else
    echo 'skipping evaluation'
fi
