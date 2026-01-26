import numpy as np
import fire
import json
from pathlib import Path
from datasets import load_dataset
from loguru import logger

def pass_at_k(n, c, k):
    """
    n: total number of samples
    c: number of passing samples
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    
    # Calculate the product: product_{i=0 to k-1} (n - c - i) / (n - i)
    # This represents the probability that all k samples are incorrect.
    res = 1.0
    for i in range(k):
        res *= (n - c - i) / (n - i)
        
    return 1.0 - res


def main(
    dataset: str = '/home/colin/code/repotune/data/eval/sbv/django.jsonl',
    eval_dir: str = 'evals/bugfix/django_e13b714/bugfix_8k_swesmith_all_qwen34i_32k_2ep_v0_ckpt1024',
) -> dict:
    ds = load_dataset('json', data_files=dataset, split='train')
    id_to_pass = {s['instance_id']: {'count': 0} for s in ds}
    logger.info(f'{len(id_to_pass) = }')

    eval_dir = Path(eval_dir)
    num_versions = 0
    for eval_version_dir in eval_dir.iterdir():
        if not eval_version_dir.name.startswith('v'):
            continue
        for eval_res_file in eval_version_dir.glob('*.json'):
            if 'preds' in eval_res_file.name:
                continue
            with eval_res_file.open('r') as f:
                eval_res = json.load(f)
            for id_r in eval_res['resolved_ids']:
                id_to_pass[id_r]['count'] += 1
            num_versions += 1
            logger.info(f'{eval_res_file = }')
            break
        else:
            logger.warning(f'No eval result found for {eval_version_dir}')

    metrics = {'num_versions': num_versions}
    for k in range(1, num_versions + 1):
        for id, passinfo in id_to_pass.items():
            passinfo[f'pass@{k}'] = pass_at_k(num_versions, passinfo['count'], k)
        metrics[f'pass@{k}'] = np.mean([passinfo[f'pass@{k}'] for passinfo in id_to_pass.values()])

    return metrics


if __name__ == "__main__":
    fire.Fire(main)
