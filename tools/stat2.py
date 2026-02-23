import json
import copy
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import fire
import yaml
from collections import defaultdict
from natsort import natsorted
from loguru import logger

@dataclass
class Stat:
    eval_dir: str = 'evals/bugfix'

    def __post_init__(self):
        self.eval_dir = Path(self.eval_dir)

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
        init_count = {
            'num_turns': [],
            'tot_completion_tokens': [],
            'tot_all_tokens': [],
        }
        qwen34i_count = copy.deepcopy(init_count)
        mix8k_count = copy.deepcopy(init_count)
        swesmith8k_count = copy.deepcopy(init_count)
        for repo_dir in self.eval_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            repo_name = repo_dir.name.split('_')[0]
            for model_dir in repo_dir.iterdir():
                if not (
                    model_dir.is_dir()
                    and (
                        model_dir.name == 'qwen34i' or
                        model_dir.name == 'bugfix_8k_swesmith_all_qwen34i_32k_2ep_v0_ckpt1024' or
                        model_dir.name.startswith(f'mix_8k_bfta_{repo_name}_qwen34i_32k_2ep')
                    )
                ):
                    continue
                if model_dir.name == 'qwen34i':
                    count_dict = qwen34i_count
                elif model_dir.name.startswith('bugfix_8k'):
                    count_dict = swesmith8k_count
                else:
                    count_dict = mix8k_count
                
                v_count = 0
                for v_dir in natsorted(model_dir.iterdir()):
                    if not (v_dir.name.startswith('v') and len(v_dir.name) == 2 and v_dir.is_dir()):
                        continue
                    v_count += 1
                    for traj_dir in v_dir.iterdir():
                        if not (traj_dir.is_dir() and traj_dir.name.startswith(repo_name)):
                            continue
                        traj_path = traj_dir / f'{traj_dir.name}.traj.json'
                        if not traj_path.exists():
                            continue
                        traj = json.loads(traj_path.read_text())
                        usage = self._count_usage(traj)
                        for k, v in usage.items():
                            count_dict[k].append(v)
                    if v_count >= 3:
                        break
            # end for model_dir
        # end for repo_dir

        stat_dict = {}
        for count_dict, name in zip([qwen34i_count, mix8k_count, swesmith8k_count], ['qwen34i', 'mix8k', 'swesmith8k']):
            for k, v in count_dict.items():
                stat_dict[f'{name}_{k}_mean'] = np.mean(v)
                stat_dict[f'{name}_{k}_std'] = np.std(v)
            
        return stat_dict


if __name__ == "__main__":
    fire.Fire(Stat)
