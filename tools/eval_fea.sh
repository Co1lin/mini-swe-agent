set -x

PORT=${PORT:-0}
MS=${MS:-unknown}
MODEL=${MODEL:-openai/$MS}
MODEL_CLASS=${MODEL_CLASS:-litellm}
REPO=${REPO:-django}
HASH=${HASH:-e13b714}
VERSION=${VERSION:-1}
CONFIG=${CONFIG:-configs/fea/gpt.yaml}
WORKERS=${WORKERS:-12}
RUN_EVAL=${RUN_EVAL:-true}

REPO_HASH=${REPO}_${HASH}
OUTPUT_DIR=evals/fea/$REPO_HASH/$MS/v$VERSION
RUN_ID=fea_${REPO_HASH}_${MS}_v${VERSION}

# /home/colin/code/repotune/data/eval/fea/$REPO.jsonl
DS=/home/tianjun/fea-bench/feabench-data/cleaned/$REPO.jsonl

sleep 3s
wc -l $DS

uv run mini-extra swebench -c $CONFIG --workers $WORKERS \
    --subset $DS \
    --model $MODEL \
    --model-class $MODEL_CLASS \
    --remote-port-selection $PORT \
    --output $OUTPUT_DIR

if [ "$RUN_EVAL" = "true" ]; then
    docker ps -aq --filter "name=$RUN_ID" | xargs -r docker rm -f
    sleep 3s

    uv run python -m feaswebench.harness.run_evaluation \
        --dataset_name $DS \
        --predictions_path $OUTPUT_DIR/preds.json \
        --timeout 600 \
        --max_workers 12 \
        --run_id $RUN_ID \
        --cache_level instance

    mv logs/run_evaluation/$RUN_ID $OUTPUT_DIR/logs
    mv *$RUN_ID.json $OUTPUT_DIR/

    # uv run tools/eval_loc.py --pred_file $OUTPUT_DIR/preds.json 2>&1 | tee -a $OUTPUT_DIR/report.log

    uv run tools/stat.py run --eval_dir $OUTPUT_DIR | tee -a $OUTPUT_DIR/report.log

    grep _instances $OUTPUT_DIR/*$RUN_ID.json | tee -a $OUTPUT_DIR/report.log
else
    echo 'skipping evaluation'
fi
