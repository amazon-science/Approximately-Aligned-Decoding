import argparse
import csv
import json
from pathlib import Path
from random import shuffle

from approx_aligned_decoding.utils import set_seed

cyrillic_lookalike_vowels = ["а", "е", "о", "А", "Е", "О"]

def run(file):
    file = Path(file)
    output_rater = file.with_name(f"{file.stem}_rater.csv")
    output_key = file.with_name(f"{file.stem}_key.json")
    set_seed(0)

    # Load all elemnts of file
    with open(file, "r") as f:
        lines = f.readlines()
        entries = [json.loads(line) for line in lines]

    indices = list(range(len(entries)))
    shuffle(indices)

    # Write rater file
    with open(output_rater, "w") as f:
        w = csv.DictWriter(f, fieldnames=["id", "Prompt", "Answer", "Score", "Follows Intent", "Rater ID",
                                          "Highlighted Cyrillic"])
        w.writeheader()
        for i, idx in enumerate(indices):
            entry = entries[idx]
            if entry["output"].endswith("</s>"):
                entry["output"] = entry["output"][:-4]

            violates_constraint = any(l in entry["output"] for l in (entry["letter"].lower(), entry["letter"].upper()))

            contains_cyrillic = any(l in entry["output"] for l in cyrillic_lookalike_vowels)
            highlighted_cyrillic = entry["output"]
            for l in cyrillic_lookalike_vowels:
                highlighted_cyrillic = highlighted_cyrillic.replace(l, f"<<{l}>>")

            w.writerow({
                "id": i,
                "Prompt": f"{entry['prompt_base']} without using the letter \"{entry['letter']}\".",
                "Answer": entry["output"],
                "Score": "",
                "Follows Intent": "X" if violates_constraint else "",
                "Highlighted Cyrillic": highlighted_cyrillic if contains_cyrillic else ""
            })

    with open(output_key, "w") as f:
        json.dump(indices, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    run(args.file)
