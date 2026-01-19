from dataclasses import dataclass
import shutil
import json
from pathlib import Path
from tqdm import tqdm

import fire
from loguru import logger

@dataclass
class FixPreds:
    input_path: str = "evals/django_e13b714/sbv_gemini3flash/preds.json"
    backup_path: str = ""  # 'evals/django_e13b714/sbv_gemini3flash/preds.bak.json'

    def __post_init__(self) -> None:
        if not self.backup_path:
            self.backup_path = Path(self.input_path[: self.input_path.rfind(".")] + ".bak.json")
        self.input_path = Path(self.input_path)

    def rm_empty(self) -> None:
        preds = json.loads(Path(self.input_path).read_text())
        Path(self.backup_path).write_text(json.dumps(preds, indent=2))

        preds_new = {k: v for k, v in preds.items() if v["model_patch"].strip()}
        logger.info(f"Removed: {len(preds) - len(preds_new)}")
        Path(self.input_path).write_text(json.dumps(preds_new, indent=2))
    
    def rm_invalid(self) -> None:
        preds = json.loads(Path(self.input_path).read_text())
        Path(self.backup_path).write_text(json.dumps(preds, indent=2))

        preds_new = {k: v for k, v in preds.items() if v["model_patch"].startswith('diff --git')}
        logger.info(f"Removed: {len(preds) - len(preds_new)}")
        Path(self.input_path).write_text(json.dumps(preds_new, indent=2))
    
    def remove_large(self) -> None:
        shutil.copyfile(self.input_path, self.backup_path)
        with self.input_path.open() as f:
            logger.info(f'loading {self.input_path}')
            preds = json.load(f)
        preds_new = {k: v for k, v in tqdm(preds.items()) if len(v['model_patch']) < 0.5 * 1e6}
        logger.info(f"Removed: {len(preds) - len(preds_new)}")
        Path(self.input_path).write_text(json.dumps(preds_new, indent=2))
    
    def gen_preds(self, traj_dir: str) -> None:
        traj_dir = Path(traj_dir)
        preds = {}
        for inst_dir in tqdm(traj_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            instance_id = inst_dir.name
            inst_path = inst_dir / f'{instance_id}.traj.json'
            if not (0 < inst_path.stat().st_size < 10 * 1024**2):
                logger.info(f'skip due to size: {inst_path}')
                continue
            traj = json.load(inst_path.open())
            preds[instance_id] = {
                'model_name_or_path': traj['info']['config']['model']['model_name'],
                'instance_id': instance_id,
                'model_patch': traj['info']['submission'],
            }
        (traj_dir / 'preds.json').write_text(json.dumps(preds, indent=2))


if __name__ == "__main__":
    fire.Fire(FixPreds)
