# Approximately-Aligned-Decoding
Official code for https://neurips.cc/virtual/2025/poster/120288

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper, but interested parties are encouraged to open an issue requesting open source community development.

## Overview

This repository contains code to perform approximately aligned decoding, using an efficient technique to resample after
an undesired
output is detected.
We refer to undesired outputs as errors, but this could be anything that is detectable automatically
(grammar, hallucinations, disallowed letters in lipograms, etc).

There are many components to this repository:

- `approx_aligned_decoding` Root of our python sampling code
    - `backtracking_strategy` Implementations of several strategies upon detecting a hallucination (constrained
      decoding, ASAp, our method)
    - `evaluation` Tools for evaluating our method- generation scripts, plus statistics collection
    - `hallucination_detector` Implementations of several detectors, including disallowing certain text sequences, and a
      network interface
    - `stopping_critieria` A somewhat simplistic implementation of interchangeable stopwords
    - `decoded_token_stream.py` A class for dealing with the complexity of tokens: multi-token characters, spaces that
      depend on nearby tokens, etc.
      Presents an interface to simply map tokens to characters in the output and vice-versa.
    - `generator.py` Implementation of main generation method; supports fewer options than huggingface generators for
      simplicity.
    - `model_with_history.py` Presents a functional (if slightly inefficient) interface for dealing with query and key
      cacheing, but also allowing backtracking in a nice way.
    - `probability_modifier_trie.py` Allows marking specific generations as hallucinations and redistributes their
      probability mass to other tokens.
- `generator_probability_smell_test` Verifies correct implementation of a generator through statistical testing
    - See README in subfolder
- `test` Unit tests for selected components
- `pyright_hallucination_detector` Typescript-based hallucination detector for Python function calls,
  using some undocumented APIs of the Pyright language server for static analysis.
  We create a small server that conforms to our hallucination detection network API.

## Paths

- There are a number of places you need to specify some paths before running!
    - Set the correct conda environment in `approx_aligned_decoding/evaluation/command_generator.py`
    - Set the path to node (we used v16)
      in `approx_aligned_decoding/hallucination_detector/python_hallucination_detector.py`
    - Set the path to a python environment with installed libraries/type hints
      in `pyright_hallucination_detector/src/run_server.ts`
        - Additionally in `pyright_hallucination_detector/test/(tests).ts` if you want to run those

## Instructions to run experiments

BigCodeBench experiments

1. Create an anaconda environment with the packages in `requirements.txt`
2. Run `npm install` inside the `pyright_hallucination_detector` folder
3. Create a second anaconda environment with the packages in
   `https://github.com/bigcode-project/bigcodebench/blob/main/Requirements/requirements-eval.txt`
    - You don't need to use this environment for evaluation, but it is needed for type hints
4. Update the paths to the newly created environment as in the above section
5. Run `approx_aligned_decoding/evaluation/command_generator.py`; this will give you a set of commands to run
    - The commands come in pairs- one to generate each set, and one to invoke the bigcodebench evaluation pipeline.
    - ⚠ The default command runs the bigcodebench evaluation script (and thus LLM-generated code) outside of a sandbox.
      Modify as necessary with your containerization setup. ⚠
    - Create the folder `evaluation` and invoke the scripts.
    - The first command in each pair should generate files of the form `modelname-strategy.jsonl` and sometimes
      `modelname-strategy_hallucinations.jsonl`
    - The second command will generate `modelname-strategy-sanitized.jsonl` and
      `modelname-strategy-sanitized_eval_results.json`.
6. Run `approx_aligned_decoding/evaluation/evaluate_bigcodebench.py`

Lipogram evaluation

1. Set up an anaconda environment with packages in `requirements.txt`
2. Run
   `PYTHONPATH=. python approx_aligned_decoding/evaluation/generate_lipograms.py --output lipograms.jsonl --device cuda:0 --num-samples 1` (
   with the device changed as necessary)
3. `approx_aligned_decoding/evaluation/lipogram_to_rater_file.py --file lipograms.py`
    - This generates `lipograms_rater.csv` and `lipograms_key.csv`
4. Send `lipograms_rater.csv` to your human raters. Put all the results into one spreadsheet,
   `lipograms_rater_scores.csv`.
    - It is okay if prompts are rated more than once, but make sure to keep the ID rows intact (will be duplicate IDs)
5. `approx_aligned_decoding/evaluation/evaluate_lipograms.py`

Simplified LLM mini-experiment

1. Set up environment
2. `generator_probability_smell_test/experiments.py`

## Random quirks of this repository

- We refer to errors and hallucinations interchangeably in many places
- We refer to [ASAp](https://arxiv.org/abs/2405.21047) as GAD, and our method (Approximately Aligned Decoding) as
  FastGAD in many places
    - Constrained Decoding became "Naive Backtracking"- some backtracking is necessary in the case where all tokens are
      invalid.
- Outputs with different strategies are intentionally identical unless an error is detected, when using the same seed
    - This allows us to measure *only* the effect of the backtracking strategy
- We couldn't re-use most of the huggingface transformers generation pipeline due to the backtracking component, so
  quite a bit is reimplemented
