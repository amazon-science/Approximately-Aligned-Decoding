import argparse
import dataclasses
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from approx_aligned_decoding.backtracking_strategy.fast_gad_backtracking import FastGADBacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.grammar_aligned_decoding_backtracking import \
    GrammarAlignedDecodingBacktrackingStrategy
from approx_aligned_decoding.backtracking_strategy.naive_backtracking import NaiveBacktrackingStrategy
from approx_aligned_decoding.generator import generate
from approx_aligned_decoding.hallucination_detector.banned_text_hallucination_detector import \
    BannedTextHallucinationDetector
from approx_aligned_decoding.hallucination_detector.combined_hallucination_detector import \
    CombinedHallucinationDetector
from approx_aligned_decoding.stopping_criteria.stop_words import StopWords
from approx_aligned_decoding.utils import set_seed

PROMPTS = [
    "Write a story",
    "Describe elephants",
    "Provide instructions to tie a tie",
    "Critique the Mona Lisa",
    "Summarize the history of artificial intelligence"
]

LETTERS = ["A", "E", "I", "O", "U"]

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
STRATEGIES = [
    ("Unconstrained", None),
    ("GAD", GrammarAlignedDecodingBacktrackingStrategy()),
    ("FastGAD", FastGADBacktrackingStrategy()),
    ("Constrained", NaiveBacktrackingStrategy())
]


def template(prompt, letter):
    return f"<s>[INST]{prompt} without using the letter \"{letter}\".[/INST]"


def run_lipogram_generator(output_path: str, device: str, num_samples: int):
    quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map=device)

    open_mode = "w"
    existing = set()
    if os.path.exists(output_path):
        open_mode = "a"
        with open(output_path, "r") as f:
            for line in f:
                data = json.loads(line)
                existing.add((data["strategy"], data["prompt_base"], data["letter"], data["gen_id"]))

    with open(output_path, open_mode) as f:
        for strategy_name, strategy in tqdm(STRATEGIES, desc="Strategies"):
            for prompt_base in tqdm(PROMPTS, desc="Prompts"):
                for letter in tqdm(LETTERS, desc="Letters"):
                    for gen_id in range(num_samples):
                        if (strategy_name, prompt_base, letter, gen_id) in existing:
                            continue

                        prompt = template(prompt_base, letter)
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)

                        detector = None

                        if strategy is not None:
                            detector = CombinedHallucinationDetector(BannedTextHallucinationDetector(letter.lower()),
                                                                     BannedTextHallucinationDetector(letter.upper()))
                            strategy.reset_state()

                        output, stats = generate(model=model,
                                                 max_new_tokens=200,
                                                 max_total_invocations=2000,
                                                 hallucination_detector=detector,
                                                 left_context="",
                                                 right_context="",
                                                 tokenizer=tokenizer,
                                                 backtracking_strategy=strategy,
                                                 temperature=0.8,
                                                 top_k=20,
                                                 # Pretty high, but it is sometimes the case that many tokens have the banned letter
                                                 stopping_criteria=StopWords(["</s>"]),
                                                 seed_func=lambda x: set_seed((gen_id, x)),
                                                 **inputs)

                        output = {
                            "prompt": prompt,
                            "strategy": strategy_name,
                            "output": tokenizer.decode(output),
                            "num_tokens": len(output),
                            "gen_id": gen_id,
                            "letter": letter,
                            "prompt_base": prompt_base,
                            **dataclasses.asdict(stats),
                        }

                        f.write(json.dumps(output) + "\n")
                        f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()
    run_lipogram_generator(args.output, args.device, args.num_samples)
