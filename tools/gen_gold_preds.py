from datasets import load_dataset
import json
from pathlib import Path

import fire
from loguru import logger

def main(
    input_path: str = '/home/colin/code/repotune/data/agentic/bugfix/swe_smith/sb_instances_o_128.jsonl',
    output_path: str = 'evals/syn/swesmith_o_128_hello_preds.json',
) -> None:
    ds = load_dataset('json', data_files=input_path, split='train')
    logger.info(f'{len(ds) = }')

    hello_patch = '''diff --git a/hello.txt b/hello.txt
new file mode 100644
index 0000000..ce01362
--- /dev/null
+++ b/hello.txt
@@ -0,0 +1 @@
+hello
'''

    preds: dict[str, dict] = {}
    for s in ds:
        preds[s['instance_id']] = {
            'model_name_or_path': 'gold',
            'instance_id': s['instance_id'],
            # 'model_patch': s['patch'],
            'model_patch': hello_patch,
        }
    logger.info(f'{len(preds) = }')

    Path(output_path).write_text(json.dumps(preds, indent=2))


if __name__ == '__main__':
    fire.Fire(main)
