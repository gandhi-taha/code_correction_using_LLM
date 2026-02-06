import argparse
import json
import os
import sys
from datetime import datetime, timezone

from humaneval import analysis, core

def main() -> None:

    parser = argparse.ArgumentParser(
        description="HumanEval Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Step 1: generate + unit test")
    p_gen.add_argument("--model-choice", choices=core.MODEL_CHOICES)

    p_judge = sub.add_parser("judge", help="Step 2: judge incorrect solutions")
    p_judge.add_argument("--model-choice", choices=core.MODEL_CHOICES)
    p_judge.add_argument("--judge-choice", choices=core.MODEL_CHOICES)
    p_judge.add_argument("--judge-id", default=core.JUDGE_ID)

    p_export = sub.add_parser("export", help="Step 3: export annotation bundles")
    p_export.add_argument("--model-choice", choices=core.MODEL_CHOICES)
    p_export.add_argument("--judge-id", default=core.JUDGE_ID)

    p_annot = sub.add_parser("annotate", help="CLI human annotation")
    p_annot.add_argument("--annotation-dir")
    p_annot.add_argument("--model-choice", choices=core.MODEL_CHOICES)
    p_annot.add_argument("--limit", type=int)

    p_extract = sub.add_parser("extract", help="Extract annotations to CSV")
    p_extract.add_argument("--annotation-dir")
    p_extract.add_argument("--model-choice", choices=core.MODEL_CHOICES)

    sub.add_parser("stats", help="Show dataset statistics")

    p_align = sub.add_parser("alignment", help="Evaluate human-AI alignment")
    p_align.add_argument("--human")
    p_align.add_argument("--ai")
    p_align.add_argument("--output", default="alignment_results.json")

    sub.add_parser("guides", help="Generate guide/custom files")

    p_all = sub.add_parser("run-all", help="Run full pipeline: generate → judge → export")
    p_all.add_argument("--model-choice", choices=core.MODEL_CHOICES)
    p_all.add_argument("--judge-choice", choices=core.MODEL_CHOICES)
    p_all.add_argument("--judge-id", default=core.JUDGE_ID)

    args = parser.parse_args()

    def init_manifest(params: dict) -> str:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(core.OUTPUT_DIR, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump({
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "params": params,
                "steps": []
            }, f, indent=2)
        return path

    def log_step(path: str, step: str, status: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        data["steps"].append({
            "step": step,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    if args.command == "generate":
        model = core.resolve_model_choice(args.model_choice, core.MODEL_NAME)
        df = core.load_or_prepare_dataset()
        core.run_codegen_humaneval(df, core.IDX_RANGE, model, core.ITER_NO,
                                    core.get_prompt_template_data_collection_claude, core.TEMPERATURE)
        core.run_unit_tests(df, core.IDX_RANGE, model, core.ITER_NO)
        core.save_dataframe_to_csv(df, core.WORK_DATASET_PATH)
        core.force_delete_directory(core.UNIT_TESTS_PATH)

    elif args.command == "judge":
        model = core.resolve_model_choice(args.model_choice, core.MODEL_NAME)
        judge = core.resolve_model_choice(args.judge_choice, core.JUDGE_MODEL_NAME)
        if not core.file_exists(core.DATASET_PATH):
            raise FileNotFoundError(f"{core.DATASET_PATH} not found")
        df = __import__("pandas").read_csv(core.DATASET_PATH)
        core.run_judge(df, core.IDX_RANGE, model, core.ITER_NO, judge, args.judge_id)

    elif args.command == "export":
        model = core.resolve_model_choice(args.model_choice, core.MODEL_NAME)
        analysis.export_to_json(model, args.judge_id)

    elif args.command == "annotate":
        model = core.resolve_model_choice(args.model_choice, core.MODEL_NAME)
        ann_dir = args.annotation_dir or core.annotation_dir(model)
        analysis.run_annotate(annotation_dir=ann_dir, limit=args.limit)

    elif args.command == "extract":
        model = core.resolve_model_choice(args.model_choice, core.MODEL_NAME)
        ann_dir = args.annotation_dir or core.annotation_dir(model)
        analysis.run_extract_annotations(annotation_dir=ann_dir)

    elif args.command == "stats":
        analysis.run_stats()

    elif args.command == "alignment":
        analysis.run_alignment_evaluation(args.human, args.ai, args.output)

    elif args.command == "guides":
        analysis.run_generate_guides()

    elif args.command == "run-all":
        manifest = init_manifest({
            "model_choice": args.model_choice,
            "judge_choice": args.judge_choice,
            "judge_id": args.judge_id
        })
        try:
            model = core.resolve_model_choice(args.model_choice, core.MODEL_NAME)
            df = core.load_or_prepare_dataset()
            core.run_codegen_humaneval(df, core.IDX_RANGE, model, core.ITER_NO,
                                        core.get_prompt_template_data_collection_claude, core.TEMPERATURE)
            core.run_unit_tests(df, core.IDX_RANGE, model, core.ITER_NO)
            core.save_dataframe_to_csv(df, core.WORK_DATASET_PATH)
            core.force_delete_directory(core.UNIT_TESTS_PATH)
            log_step(manifest, "generate", "ok")

            judge = core.resolve_model_choice(args.judge_choice, core.JUDGE_MODEL_NAME)
            df = __import__("pandas").read_csv(core.DATASET_PATH)
            core.run_judge(df, core.IDX_RANGE, model, core.ITER_NO, judge, args.judge_id)
            log_step(manifest, "judge", "ok")

            analysis.export_to_json(model, args.judge_id)
            log_step(manifest, "export", "ok")
        except Exception as e:
            log_step(manifest, "run-all", f"error: {e}")
            raise

if __name__ == "__main__":
    main()