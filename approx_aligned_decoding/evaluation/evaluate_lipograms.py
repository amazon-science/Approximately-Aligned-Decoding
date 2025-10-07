import csv
import json
from typing import List

import pandas as pd


def collate_entries(key: List[int], entries, rater):
    # Correlate rater scores with their original generations
    # key[0] = 40 -> rater with id 0 corresponds to entries[40]
    for src, dest in enumerate(key):
        rater_for_dest = []
        for r in rater:
            if int(r["id"]) == src:
                # Make sure the rater file matches, up to minor trimming of </s> and whitespace
                assert entries[dest]["output"].startswith(r["Answer"])
                assert len(entries[dest]["output"]) < len(r["Answer"]) + 10

                if r["Score"] == "":
                    continue

                intent = r["Follows Intent"]
                if intent == "X":
                    intent = 1

                # Sometimes the rater overrides the "X" (banned letter appears in output); turn it back into a 1
                banned_letter: str = entries[dest]["letter"]
                if banned_letter.lower() in r["Answer"] or banned_letter.upper() in r["Answer"]:
                    intent = 1

                intent = int(intent)

                rater_scores = {
                    "score": int(r["Score"]),
                    "intent": intent,
                    "rater_id": r.get("Rater ID")
                }
                rater_for_dest.append(rater_scores)

        entries[dest]["rater"] = rater_for_dest


def load_data():
    with open("lipograms_key.json", "r") as f:
        key = json.load(f)

    with open("lipograms.jsonl", "r") as f:
        entries = [json.loads(line) for line in f]

    with open("lipograms_rater_scores.csv", "r") as f:
        rater = list(csv.DictReader(f))

    collate_entries(key, entries, rater)
    return entries


def save_collated(entries):
    with open("lipograms_collated.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=entries[0].keys())
        writer.writeheader()
        for entry in entries:
            rater_output = []
            for rater_entry in entry["rater"]:
                rater_output.append(
                    f'{rater_entry.get("rater_id", "?")}:{rater_entry["score"]}/{rater_entry["intent"]}')

            writer.writerow({
                **entry,
                "rater": "|".join(rater_output)
            })


def process_data(entries):
    entry_raters = []
    for seq_id, entry in enumerate(entries):
        for rater_entry in entry["rater"]:
            entry_raters.append({
                "id": seq_id,
                "rater": rater_entry["rater_id"],
                "score": rater_entry["score"],
                "intent": rater_entry["intent"],
                "strategy": entry["strategy"],
                "gen_ratio": entry["total_generated_toks_model"] / entry["num_tokens"]
            })

    df = pd.DataFrame(entry_raters)
    dfd = df.groupby("strategy").describe()

    # Print nice latex table
    for strategy, row in dfd.iterrows():
        strategy_name = {
            "Unconstrained": "Unconstrained",
            "GAD": "ASAp",
            "FastGAD": "Ours",
            "Constrained": "Constrained"
        }[strategy]

        def metric(name):
            return f'{row[(name, "mean")]:.2f} $\\pm$ {row[(name, "std")]:.2f}'

        print(f'{strategy_name} & {metric("score")} & {metric("intent")} & {metric("gen_ratio")} \\\\')


def run():
    entries = load_data()
    save_collated(entries)
    process_data(entries)


if __name__ == '__main__':
    run()
