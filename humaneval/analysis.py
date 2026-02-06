import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

def export_to_json(model: str, judge_id: str) -> None:
    from humaneval import core

    if not core.file_exists(core.DATASET_PATH):
        raise FileNotFoundError(f"{core.DATASET_PATH} not found")

    df = pd.read_csv(core.DATASET_PATH)
    storage = core.annotation_dir(model)

    logger.info(f"Cleaning: {storage}")
    core.force_delete_directory(storage)
    os.makedirs(storage, exist_ok=True)

    count = 0
    rd_cols = [f"analysis_{model}_wt_{judge_id}_rd{i}" for i in (1, 2, 3)]
    custom_col = f"analysis_{model}_custom"

    for idx, row in df.iterrows():
        avail = [c for c in rd_cols if c in df.columns]
        if not avail or pd.isna(row.get(rd_cols[0])):
            continue

        for iter_no in core.ITER_NUMS if hasattr(core, 'ITER_NUMS') else [1]:
            res_col, eval_col = f"result_{model}_no{iter_no}", f"eval_{model}_no{iter_no}"
            if res_col not in df.columns or eval_col not in df.columns or row[eval_col] != "INCORRECT":
                continue

            gt = (row["prompt"] + row["canonical_solution"]).strip()
            buggy = (row["prompt"] + core.indent_lines(row[res_col])).strip()

            def get_val(c):
                return row[c] if c in df.columns and not pd.isna(row[c]) else "N/A"

            analyses = [(c, get_val(c)) for c in avail]
            custom_txt = get_val(custom_col)
            if custom_col in df.columns and not pd.isna(row.get(custom_col)):
                analyses.append((custom_col, custom_txt))

            task_id_raw = row["task_id"]
            if isinstance(task_id_raw, str) and "/" in task_id_raw:
                task_id_num = int(task_id_raw.split("/")[-1])
            else:
                task_id_num = int(task_id_raw)

            bundle = {
                "task_id": task_id_num,
                "ground_truth_solution": gt,
                "buggy_solution": buggy,
                "custom_analysis_reference": custom_txt,
                "analyses": [
                    {
                        "analysis_id": i + 1,
                        "col_name": src,
                        "target_analysis": txt,
                        "scores": {f"S{j}": ("7" if src == custom_col else "_") for j in range(1, 7)}
                    }
                    for i, (src, txt) in enumerate(analyses)
                ]
            }

            fpath = os.path.join(storage, f"{idx}.json")
            with open(fpath, "w") as f:
                json.dump(bundle, f, ensure_ascii=False, indent=2)
            count += 1
            break

    logger.info(f"Exported {count} tasks to {storage}")

def run_stats() -> None:
    from humaneval import core

    if not core.file_exists(core.DATASET_PATH):
        logger.warning(f"{core.DATASET_PATH} not found")
        return

    df = pd.read_csv(core.DATASET_PATH)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    for col in df.columns:
        if col.startswith("eval_"):
            logger.info(f"\n{col}:")
            print(df[col].value_counts())

def run_annotate(annotation_dir: str, limit: Optional[int] = None) -> None:
    if not os.path.isdir(annotation_dir):
        logger.error(f"Annotation directory not found: {annotation_dir}")
        return

    json_files = sorted(Path(annotation_dir).glob("*.json"))
    if limit:
        json_files = json_files[:limit]

    logger.info(f"Found {len(json_files)} annotation files")

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        logger.info(f"\n=== Task {data['task_id']} ===")
        logger.info(f"Buggy solution:\n{data['buggy_solution']}")
        logger.info(f"\nAnalyses:")
        for a in data['analyses']:
            logger.info(f"  [{a['analysis_id']}] {a['col_name']}: {a['target_analysis'][:100]}...")

def run_extract_annotations(annotation_dir: str) -> None:
    if not os.path.isdir(annotation_dir):
        logger.error(f"Annotation directory not found: {annotation_dir}")
        return

    json_files = list(Path(annotation_dir).glob("*.json"))
    logger.info(f"Extracting from {len(json_files)} annotation files")

def run_alignment_evaluation(human_file: str, ai_file: str, output_file: str) -> None:
    logger.info("Alignment evaluation placeholder")

def run_generate_guides() -> None:
    logger.info("Guide generation placeholder")