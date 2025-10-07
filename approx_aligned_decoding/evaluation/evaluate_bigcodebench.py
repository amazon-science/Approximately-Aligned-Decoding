import argparse
import json

import datasets
import pandas as pd

from approx_aligned_decoding.evaluation.command_generator import models, strategies, get_base_name
from approx_aligned_decoding.evaluation.generate_bigcodebench import add_imports_from_test


def is_name_error(col: pd.Series):
    def per_row(details):
        return any(any(x in d for x in ("NameError", "UnboundLocalError")) for d in details.values())

    return col.apply(per_row)


def pass_at_one_five(col: pd.Series):
    pass_at_1 = (col / 5).mean()
    pass_at_5 = (col >= 1).sum() / len(col)
    return pass_at_1, pass_at_5


def run_evaluations(entries, eval_results):
    # Correctness results
    eval_results["name_error"] = is_name_error(eval_results["details"])
    eval_results["not_name_error"] = ~eval_results["name_error"]
    eval_results["is_pass"] = eval_results["status"] == "pass"
    num_pass_per_task = eval_results.groupby("task_id").agg({"is_pass": "sum", "not_name_error": "sum"})
    pass_at_1, pass_at_5 = pass_at_one_five(num_pass_per_task["is_pass"])
    not_ne_at_1, not_ne_at_5 = pass_at_one_five(num_pass_per_task["not_name_error"])

    # Performance results
    generation_ratio = entries["total_generated_toks_model"] / entries["num_tokens"]
    mean_generation_ratio = generation_ratio.mean()
    std_generation_ratio = generation_ratio.std()

    print(f"Pass@1: {pass_at_1}, Pass@5: {pass_at_5}")
    print(f"Not NameError@1: {not_ne_at_1}, Not NameError@5: {not_ne_at_5}")
    print(f"Mean Generation Ratio: {mean_generation_ratio}, Std Generation Ratio: {std_generation_ratio}")

    # Row to copy-paste into latex
    latex_code = (
        f"{pass_at_1:.3f} & {pass_at_5:.3f} & {not_ne_at_1:.3f} & {not_ne_at_5:.3f} & {mean_generation_ratio:.3f} $\\pm$ {std_generation_ratio:.3f} \\\\")

    return latex_code, {
        "pass@1": pass_at_1,
        "pass@5": pass_at_5,
        "not_name_error@1": not_ne_at_1,
        "not_name_error@5": not_ne_at_5,
        "mean_generation_ratio": mean_generation_ratio,
        "std_generation_ratio": std_generation_ratio
    }


bcb = datasets.load_dataset("bigcode/bigcodebench")["v0.1.0_hf"]


def load_data(base_name):
    output_name = base_name + ".jsonl"
    eval_name = base_name + "-sanitized_eval_results.json"

    with open(output_name, "r") as f:
        lines = f.readlines()
        entries = [json.loads(line) for line in lines]
        for entry in entries:
            entry_id = int(entry["task_id"].split("/")[-1])
            b = bcb[entry_id]
            entry_prompt = add_imports_from_test(b["complete_prompt"], b["test"])
            entry["completion"] = entry["solution"][len(entry_prompt.strip()):]
        entries = pd.DataFrame(entries).sort_index()

    with open(eval_name, "r") as f:
        eval_results = json.load(f)["eval"]

        data = {}
        for task_id, evaluations in eval_results.items():
            for i, evaluation in enumerate(evaluations):
                data[(task_id, i)] = evaluation

        eval_results = pd.DataFrame(data).transpose().sort_index()

    return entries, eval_results


def run(only_diff_entries):
    for model in models:
        data = []

        for strategy_name, _ in strategies:
            basename = get_base_name(model, strategy_name)
            entries, eval_results = load_data(basename)
            data.append((strategy_name, entries, eval_results))

        if only_diff_entries:
            identical = (data[0][1]["solution"] == data[1][1]["solution"])
            for i in range(2, len(data)):
                identical = identical & (data[0][1]["solution"] == data[i][1]["solution"])

            entries_with_identical = data[0][1].copy()
            entries_with_identical["identical"] = identical

            num_identical_by_task_id = entries_with_identical.groupby("task_id").agg({"identical": "sum"})
            num_identical_by_task_id["any_not_identical"] = num_identical_by_task_id["identical"] < 5

            print(
                f"Num completions with different outputs: {int((~identical).sum())}/{len(identical)} ({float((~identical).sum() / len(identical))})")
            print(
                f"Num tasks with different outputs: {int(num_identical_by_task_id['any_not_identical'].sum())}/{len(num_identical_by_task_id)} ({float(num_identical_by_task_id['any_not_identical'].sum() / len(num_identical_by_task_id))})")
            print("\n")

        latex_outputs = []

        for strategy_name, entries, eval_results in data:
            print(f"{model} - {strategy_name}")

            if only_diff_entries:
                entries = entries.join(num_identical_by_task_id, on="task_id")
                entries = entries[entries["any_not_identical"]]

                eval_results = eval_results.join(num_identical_by_task_id, on="task_id")
                eval_results = eval_results[eval_results["any_not_identical"]]

            latex_code, _ = run_evaluations(entries, eval_results)
            latex_outputs.append((strategy_name, latex_code))
            print("\n")

        print(f"\\multirow{{{len(latex_outputs)}}}{{*}}{{{model}}}", end="")
        for strategy_name, latex_code in latex_outputs:
            strat_nice = {
                "gad": "ASAp",
                "fastgad": "Ours",
                "naive": "Constrained",
                "unconstrained": "Unconstrained"
            }[strategy_name]
            print(f"& {strat_nice} & {latex_code}\\")

        print("")

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--only-diff-entries', action="store_true")
    args = a.parse_args()
    run(args.only_diff_entries)
