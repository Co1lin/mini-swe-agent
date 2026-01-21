set -x

PORT=${PORT:-0}
MS=${MS:-unknown}
MODEL=${MODEL:-openai/$MS}
MODEL_CLASS=${MODEL_CLASS:-litellm}
REPO=${REPO:-django}
HASH=${HASH:-e13b714}
VERSION=${VERSION:-0}
CONFIG=${CONFIG:-configs/tdd/host.yaml}
WORKERS=${WORKERS:-12}
RUN_EVAL=${RUN_EVAL:-true}

REPO_HASH=${REPO}_${HASH}
OUTPUT_DIR=evals/tdd/$REPO_HASH/$MS/v$VERSION
RUN_ID=tdd_${REPO_HASH}_${MS}_v${VERSION}

uv run mini-extra swebench -c $CONFIG --workers $WORKERS \
    --subset ../repotune/data/eval/tdd/$REPO.jsonl \
    --model $MODEL \
    --model-class $MODEL_CLASS \
    --remote-port-selection $PORT \
    --output $OUTPUT_DIR

if [ "$RUN_EVAL" = "true" ]; then
    docker ps -aq --filter "name=$RUN_ID" | xargs -r docker rm -f
    sleep 3s

    uv run python -m tddbench.harness.run_evaluation \
        --dataset_name ../repotune/data/eval/tdd/$REPO.jsonl \
        --predictions_path $OUTPUT_DIR/preds.json \
        --max_workers 16 \
        --run_id $RUN_ID \
        --cache_level instance
    
    mv logs/run_evaluation/$RUN_ID $OUTPUT_DIR/logs
    mv *$RUN_ID.json $OUTPUT_DIR/

    uv run tools/stat.py run --eval_dir $OUTPUT_DIR | tee -a $OUTPUT_DIR/report.log
    grep '"resolved": true' -r $OUTPUT_DIR/logs | wc -l | tee -a $OUTPUT_DIR/report.log
else
    echo 'skipping evaluation'
fi
