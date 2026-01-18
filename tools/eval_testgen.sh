set -x

PORT=${PORT:-0}
MS=${MS:-unknown}
MODEL=${MODEL:-openai/$MS}
MODEL_CLASS=${MODEL_CLASS:-litellm}
REPO=${REPO:-django}
HASH=${HASH:-e13b714}
VERSION=${VERSION:-0}
CONFIG=${CONFIG:-configs/testgen/host.yaml}
WORKERS=${WORKERS:-12}
RUN_EVAL=${RUN_EVAL:-true}

REPO_HASH=${REPO}_${HASH}
OUTPUT_DIR=evals/testgen/$REPO_HASH/$MS/v$VERSION
RUN_ID=testgen_${REPO_HASH}_${MS}_v${VERSION}

uv run mini-extra swebench -c $CONFIG --workers $WORKERS \
    --subset ../repotune/data/eval/swt/$REPO.jsonl \
    --model $MODEL \
    --model-class $MODEL_CLASS \
    --remote-port-selection $PORT \
    --output $OUTPUT_DIR

if [ "$RUN_EVAL" = "true" ]; then
    uv run python -m swtbench.main \
        --dataset_name ../repotune/data/eval/swt/$REPO.jsonl \
        --predictions_path $OUTPUT_DIR/preds.json \
        --max_workers 12 \
        --run_id $RUN_ID \
        --skip_gold True \
        --compute_coverage False \
        --cache_level instance

    mv evaluation_results/*$RUN_ID.json $OUTPUT_DIR/
    mv run_instance_swt_logs/$RUN_ID $OUTPUT_DIR/log_$RUN_ID

    # uv run tools/eval_loc.py --pred_file $OUTPUT_DIR/preds.json 2>&1 | tee -a $OUTPUT_DIR/report.log

    # uv run tools/stat.py run --eval_dir $OUTPUT_DIR | tee -a $OUTPUT_DIR/report.log

    grep _instances $OUTPUT_DIR/*$RUN_ID.json | tee -a $OUTPUT_DIR/report.log
else
    echo 'skipping evaluation'
fi
