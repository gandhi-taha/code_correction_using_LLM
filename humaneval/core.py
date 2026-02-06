import ast
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Dict, Optional, Tuple

import litellm
import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for _lg in ("litellm", "LiteLLM"):
    logging.getLogger(_lg).setLevel(logging.WARNING)

MODEL_CATALOG = {
    "haiku35": "anthropic/claude-3-5-haiku-20241022",
    "haiku45": "anthropic/claude-haiku-4-5-20251001",
    "haiku3": "anthropic/claude-3-haiku-20240307",
    "sonnet35": "anthropic/claude-3-5-sonnet-20240620",
}
MODEL_CHOICES = sorted(MODEL_CATALOG.keys())

MODEL_NAME = "anthropic/claude-3-haiku-20240307"
JUDGE_MODEL_NAME = "anthropic/claude-3-haiku-20240307"
JUDGE_ID = "cl3-haiku-v1"
TEMPERATURE = 0.0
ITER_NO = 1
IDX_RANGE = (0, 164)
NUM_ROUNDS = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
UNIT_TESTS_PATH = os.path.join(PROJECT_ROOT, "humaneval_unit_tests")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "humaneval_eval_dataset.csv")
WORK_DATASET_PATH = os.path.join(DATA_DIR, "humaneval_eval_dataset.csv")

load_dotenv()
litellm.suppress_instrumentation = True
os.environ["LITELLM_LOG"] = "ERROR"

def file_exists(path: str) -> bool:
    return os.path.isfile(path)

def create_directories(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def save_string_to_file(path: str, content: str) -> None:
    create_directories(path)
    with open(path, "w") as f:
        f.write(content)

def force_delete_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
        logger.info(f"Deleted: {path}")

def save_dataframe_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    create_directories(path)
    df.to_csv(path, index=index)
    logger.info(f"Saved: {path}")

def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "_")

def annotation_dir(model_id: str) -> str:
    return os.path.join(OUTPUT_DIR, f"analysis_{safe_model_name(model_id)}_human_annotation_dataset")

def custom_text_dir(model_id: str) -> str:
    return os.path.join(OUTPUT_DIR, f"analysis_{safe_model_name(model_id)}_custom_dataset")

def resolve_model_choice(choice: Optional[str], default: str) -> str:
    if not choice:
        return default
    if choice not in MODEL_CATALOG:
        raise ValueError(f"Unknown model: {choice}")
    return MODEL_CATALOG[choice]

def load_or_prepare_dataset() -> pd.DataFrame:
    if file_exists(WORK_DATASET_PATH):
        return pd.read_csv(WORK_DATASET_PATH)
    if file_exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    logger.info("Downloading HumanEval from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval")["test"]
        df = pd.DataFrame([{
            "task_id": idx,
            "prompt": d["prompt"],
            "test": d["test"],
            "entry_point": d["entry_point"],
            "canonical_solution": d["canonical_solution"]
        } for idx, d in enumerate(ds)])
        save_dataframe_to_csv(df, DATASET_PATH)
        return df
    except Exception as e:
        logger.error(f"Failed to load HumanEval: {e}")
        raise

def get_prompt_template_data_collection_claude(prompt: str) -> str:
    return f"""<instructions>
  <bullets>
    <bullet>Only include code following the docstring.</bullet>
    <bullet>Do not include function header or comments.</bullet>
  </bullets>
</instructions>

<docstring>
{prompt}
</docstring>"""

def get_prompt_template_judge_claude(ground_truth: str, buggy: str, error: Optional[str] = None) -> str:
    error_section = f"\n<error_message>\n{error}\n</error_message>\n" if error else ""
    return f"""<instructions>
  <bullets>
    <bullet>Ground truth is the correct implementation.</bullet>
    <bullet>Buggy code contains one or more bugs.</bullet>
    <bullet>Find all bugs and quote each buggy section.</bullet>
    <bullet>Describe each bug and outline fix using ground truth.</bullet>
    <bullet>Do not explicitly mention ground truth code.</bullet>
    <bullet>Cover all existing bugs, be concise.</bullet>
  </bullets>
</instructions>

<ground_truth_code>
{ground_truth}
</ground_truth_code>

<buggy_code>
{buggy}
</buggy_code>{error_section}"""

def indent_lines(s: str) -> str:
    return "\n".join("    " + line for line in s.splitlines())

def contains_function_definition(code: str) -> bool:
    try:
        return any(isinstance(node, ast.FunctionDef) for node in ast.walk(ast.parse(code)))
    except SyntaxError:
        return False

def slashes_to_underscores(s: str) -> str:
    return s.replace("/", "_")

def get_completion(prompt: str, model: str, temperature: Optional[float] = None, max_retries: int = 10) -> str:
    args = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "verbose": False
    }
    if temperature is not None:
        args["temperature"] = temperature

    for attempt in range(max_retries):
        try:
            with open(os.devnull, "w") as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
                resp = completion(**args)
            return resp.choices[0].message.content
        except (litellm.exceptions.InternalServerError, litellm.exceptions.RateLimitError,
                litellm.exceptions.ServiceUnavailableError, litellm.exceptions.APIConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.random())
            else:
                raise e
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.random())
            else:
                raise e

def run_python_file(path: str, timeout: int = 10) -> Tuple[str, str]:
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return ("CORRECT", "")
        error_msg = r.stderr[:2000] if r.stderr else ""
        return ("INCORRECT", error_msg)
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout ({timeout}s): {path}")
        return ("INVALID", f"Timeout: execution exceeded {timeout} seconds")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return ("INVALID", "File not found")
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        return ("INVALID", f"Error: {str(e)[:200]}")

def run_codegen_humaneval(df: pd.DataFrame, idx_range: Tuple[int, int], model: str,
                          iter_no: int, get_prompt: Callable[[str], str], temperature: Optional[float] = None) -> Dict[str, int]:
    start, end = idx_range
    length = len(df)
    if start >= length:
        logger.warning("Index range beyond dataset; skipping codegen.")
        return {"total": 0, "valid": 0, "error": 0}

    end = min(end, length)
    col = f"result_{model}_no{iter_no}"
    if col not in df.columns:
        df[col] = None

    stats = {"total": 0, "valid": 0, "error": 0}
    for idx in tqdm(range(start, end), desc="Codegen"):
        try:
            prompt = df.iloc[idx]["prompt"]
            df.at[idx, col] = get_completion(get_prompt(prompt), model, temperature)
            stats["valid"] += 1
        except Exception as e:
            logger.error(f"Task {idx}: {e}")
            stats["error"] += 1
        stats["total"] += 1

    return stats

def run_unit_tests(df: pd.DataFrame, idx_range: Tuple[int, int], model: str, iter_no: int) -> Dict[str, int]:
    start, end = idx_range
    if start >= len(df):
        logger.warning("Index range beyond dataset; skipping tests.")
        return {"total": 0, "correct": 0, "incorrect": 0, "invalid": 0, "timeout": 0}

    end = min(end, len(df))
    eval_col, error_col = f"eval_{model}_no{iter_no}", f"error_{model}_no{iter_no}"

    for c in (eval_col, error_col):
        if c not in df.columns:
            df[c] = "" if c == error_col else None
    if df[error_col].dtype != object:
        df[error_col] = df[error_col].astype(str)

    stats = {"total": 0, "correct": 0, "incorrect": 0, "invalid": 0, "timeout": 0}

    for idx in tqdm(range(start, end), desc="Unit Tests"):
        result = df.iloc[idx][f"result_{model}_no{iter_no}"]
        if contains_function_definition(result):
            df.at[idx, eval_col] = "INVALID"
            df.at[idx, error_col] = "Contains function definition"
            stats["invalid"] += 1
            stats["total"] += 1
            continue

        row = df.iloc[idx]
        code = row["prompt"] + indent_lines(result) + "\n\n" + row["test"]
        code += f"\n\ntry:\n  check({row['entry_point']})\n  exit(0)\nexcept:\n  exit(1)\n"

        fpath = f"humaneval_unit_tests/{slashes_to_underscores(model)}/{iter_no}/{idx}.py"
        save_string_to_file(fpath, code)

        outcome, err = run_python_file(fpath, timeout=10)
        df.at[idx, eval_col] = outcome
        df.at[idx, error_col] = err or ""

        stats["total"] += 1
        if outcome == "CORRECT":
            stats["correct"] += 1
        elif outcome == "INCORRECT":
            stats["incorrect"] += 1
        elif "Timeout" in err:
            stats["timeout"] += 1
        else:
            stats["invalid"] += 1

    logger.info(f"Stats {model}: Total={stats['total']}, Correct={stats['correct']}, "
                f"Incorrect={stats['incorrect']}, Invalid={stats['invalid']}, Timeout={stats['timeout']}")
    return stats

def run_judge(df: pd.DataFrame, idx_range: Tuple[int, int], model: str, iter_no: int, judge_model: str, judge_id: str) -> None:
    start, end = idx_range
    if start >= len(df):
        return

    end = min(end, len(df))
    cols = [f"analysis_{model}_wt_{judge_id}_rd{r}" for r in range(1, NUM_ROUNDS + 1)]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        res_col, eval_col = f"result_{model}_no{iter_no}", f"eval_{model}_no{iter_no}"
        if res_col not in df.columns or eval_col not in df.columns or row[eval_col] != "INCORRECT":
            continue

        err_col = f"error_{model}_no{iter_no}"
        err = row[err_col] if err_col in df.columns else None
        gt = (row["prompt"] + row["canonical_solution"]).strip()
        buggy = (row["prompt"] + indent_lines(row[res_col])).strip()
        prompt = get_prompt_template_judge_claude(gt, buggy, err)

        logger.info(f"Critiquing task {row['task_id']}...")
        for c in cols:
            df.at[idx, c] = get_completion(prompt, judge_model)

    save_dataframe_to_csv(df, DATASET_PATH)