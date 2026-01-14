import json
import copy
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import fire
import yaml
from collections import defaultdict

from loguru import logger

def check_repetition(traj_path: Path) -> bool:
    traj = json.loads(traj_path.read_text())
    c_appearances = {}
    for m in traj["messages"]:
        if m["role"] != "assistant":
            continue
        content = m["content"].strip()
        c_appearances[content] = c_appearances.get(content, 0) + 1

    return any(v >= 3 for v in c_appearances.values())

@dataclass
class Stat:
    eval_dir: str = 'evals/django_e13b714/syn_bugfix_django_qwen34i_32k_3ep_v0/ckpt639_64k_rp'

    def __post_init__(self):
        self.eval_dir = Path(self.eval_dir)
        # build self.instances_by_status
        self.instances_by_status = self._build_instances_by_status()
        self.global_num_instances = sum(len(instance_ids) for instance_ids in self.instances_by_status.values())
        print(f"{self.global_num_instances = }")

    def _build_instances_by_status(self) -> dict:
        # 1. find and load the file eval_dir / exit_statuses_*.yaml
        status_files = list(self.eval_dir.glob("exit_statuses_*.yaml"))
        if not status_files:
            raise FileNotFoundError(f"No exit_statuses_*.yaml found in {self.eval_dir}")

        instances_by_exit_status: dict[str, list[str]] = defaultdict(list)
        for status_file in status_files:
            data = yaml.safe_load(status_file.open()).get("instances_by_exit_status", {})
            for status, instance_ids in data.items():
                instances_by_exit_status[status] = list(set(instances_by_exit_status[status] + instance_ids))
            
        # Use the first one found
        # status_file = status_files[0]
        # with open(status_file) as f:
        #     data = yaml.safe_load(f)

        return instances_by_exit_status
    
    def get_traj(self, instance_id: str) -> dict:
        traj_path = self.eval_dir / instance_id / f"{instance_id}.traj.json"
        return json.loads(traj_path.read_text())

    def _check_repetition(self, traj: dict) -> bool:
        c_appearances = {}
        for m in traj["messages"]:
            if m["role"] != "assistant":
                continue
            content = m["content"].strip()
            c_appearances[content] = c_appearances.get(content, 0) + 1

        return any(v >= 3 for v in c_appearances.values())
    
    def repetition(self) -> dict:
        results = {}
        for status, instance_ids in self.instances_by_status.items():
            if not instance_ids:
                results[status] = "0 / 0"
                continue

            repetition_count = 0
            total_instances = len(instance_ids)

            for instance_id in instance_ids:
                traj = self.get_traj(instance_id)
                if self._check_repetition(traj):
                    repetition_count += 1

            results[status] = f"{repetition_count} / {total_instances}"
        
        print("######## Repetition results:")
        for status, count in results.items():
            print(f"{status}: {count}")
        print("########")
        
        return results
    
    def _count_usage(self, traj: dict) -> dict:
        tot_completion_tokens = tot_all_tokens = 0
        num_turns = 0
        for m in traj['messages']:
            if m['role'] == 'assistant':
                num_turns += 1
                usage = m['extra']['response']['usage']
                tot_completion_tokens += usage['completion_tokens']
                tot_all_tokens = usage['total_tokens']

        return {
            'num_turns': num_turns,
            'tot_completion_tokens': tot_completion_tokens,
            'tot_all_tokens': tot_all_tokens
        }

    def usage(self) -> dict:
        results = {}
        init_count = {
            'num_turns': [],
            'tot_completion_tokens': [],
            'tot_all_tokens': [],
        }
        for status, instance_ids in self.instances_by_status.items():
            if not instance_ids:
                results[status] = copy.deepcopy(init_count)
                continue

            for instance_id in instance_ids:
                traj = self.get_traj(instance_id)
                usage = self._count_usage(traj)
                for k, v in usage.items():
                    results.setdefault(status, copy.deepcopy(init_count))[k].append(v)
        

        all_status_count = copy.deepcopy(init_count)
        for status, count in results.items():
            for k, v in count.copy().items():
                all_status_count[k].extend(v)
        results['all'] = all_status_count

        for status, count in results.items():
            for k, v in count.copy().items():
                results[status].update({
                    f'{k}_mean_': np.mean(v),
                    f'{k}_median_': np.median(v),
                })
        
        print("######## Usage results:")
        for status, count in results.items():
            if status not in ['all', 'Submitted']:
                continue
            for k, v in count.items():
                if k.endswith('_'):
                    print(f"{status}: {k}: {v:.1f}")
        print(f'-------- all/mean:')
        for k, v in results['all'].items():
            if k.endswith('_mean_'):
                k_show = {
                    'num_turns_mean_': 'Turns',
                    'tot_completion_tokens_mean_': 'Compl',
                    'tot_all_tokens_mean_': 'All',
                }[k]
                print(f"{k_show}: {v:.1f}")
        print("########")
        
        return results

    def run(self) -> None:
        self.repetition()
        self.usage()


if __name__ == "__main__":
    fire.Fire(Stat)
