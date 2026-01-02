import json
from pathlib import Path

import fire
import yaml


def check_repetition(traj_path: Path) -> bool:
    traj = json.loads(traj_path.read_text())
    c_appearances = {}
    for m in traj["messages"]:
        if m["role"] != "assistant":
            continue
        content = m["content"].strip()
        c_appearances[content] = c_appearances.get(content, 0) + 1

    return any(v >= 3 for v in c_appearances.values())


def main(
    eval_dir: str = "evals/django_e13b714/syn_bugfix_django_qwen34i_32k_3ep_v0/ckpt639_64k_rp",
) -> dict:
    eval_path = Path(eval_dir)

    # 1. find and load the file eval_dir / exit_statuses_*.yaml
    status_files = list(eval_path.glob("exit_statuses_*.yaml"))
    if not status_files:
        raise FileNotFoundError(f"No exit_statuses_*.yaml found in {eval_dir}")

    # Use the first one found
    status_file = status_files[0]
    with open(status_file) as f:
        data = yaml.safe_load(f)

    # 2. in instances_by_exit_status, get the instance ids grouped by exit statuses
    instances_by_status = data.get("instances_by_exit_status", {})
    results = {}
    global_num_instances = 0

    for status, instance_ids in instances_by_status.items():
        if not instance_ids:
            results[status] = "0 / 0"
            continue

        repetition_count = 0
        total_instances = len(instance_ids)
        global_num_instances += total_instances

        for instance_id in instance_ids:
            # 3. Path construction: eval_dir / instance_id / instance_id.traj.json
            traj_path = eval_path / instance_id / f"{instance_id}.traj.json"

            # 4. check each traj.json file for repetition
            if check_repetition(traj_path):
                repetition_count += 1

        # 5. report f'{num_repetitions} / {num_instances}' for each exit status
        results[status] = f"{repetition_count} / {total_instances}"

    results["global_num_instances"] = global_num_instances

    return results


if __name__ == "__main__":
    fire.Fire(main)
