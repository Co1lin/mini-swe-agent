import json
from pathlib import Path

import fire
from loguru import logger


def main(
    input_path: str = "evals/django_e13b714/sbv_gemini3flash/preds.json",
    backup_path: str = "",  # 'evals/django_e13b714/sbv_gemini3flash/preds.bak.json',
) -> None:
    if not backup_path:
        backup_path = input_path[: input_path.rfind(".")] + ".bak.json"
    preds = json.loads(Path(input_path).read_text())
    Path(backup_path).write_text(json.dumps(preds, indent=2))

    preds_new = {k: v for k, v in preds.items() if v["model_patch"].strip()}
    logger.info(f"Removed: {len(preds) - len(preds_new)}")
    Path(input_path).write_text(json.dumps(preds_new, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
