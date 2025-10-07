import argparse
from collections import Counter
from time import sleep
from typing import Iterator

import datasets

from approx_aligned_decoding.evaluation.utils import add_standard_args, setup_env_for_generation, \
    GenerationInput, GenerationOutput, do_generation_dataset
from approx_aligned_decoding.hallucination_detector.python_hallucination_detector import \
    create_hallucination_detection_server
from approx_aligned_decoding.stopping_criteria.stop_words import StopWords

stop_words = ["<|endoftext|>",
              "<|endofmask|>",
              "<file_sep>",
              "</s>",
              "\nif",
              "\nfor",
              "\ndef",
              "\nwith",
              "\nprint",
              "\nclass",
              "\nimport",
              "\nfrom",
              "\nassert",
              "\nasync",
              "\nawait",
              "\ntry",
              ]


def extract_imports(code: str):
    lines = code.splitlines(keepends=True)
    resulting_imports = []
    remaining_code = []
    for line in lines:
        if line.startswith("import") or line.startswith("from"):
            resulting_imports.append(line)
        else:
            remaining_code.append(line)
    return resulting_imports, remaining_code


def add_imports_from_test(code: str, test: str):
    # Imports in the test are available when evaluating the code
    # If it would pass without a name error, we want it to be available to the language server
    # So we add any imports seen in the test cases

    imports_already_in_prompt, _ = extract_imports(code)
    imports_in_test = [im for im in extract_imports(test)[0] if
                       (im not in imports_already_in_prompt and "unittest" not in im)]

    # Bigcodebench harness always makes these available to programs
    special_imports = [
        "import os\n",
        "import sys\n",
        "from os import environ\n"
    ]
    special_imports = [im for im in special_imports if
                       im not in imports_in_test and im not in imports_already_in_prompt]

    return "".join(imports_in_test) + "".join(special_imports) + code


if __name__ == '__main__':
    setup_env_for_generation()

    parser = argparse.ArgumentParser()
    add_standard_args(parser)
    parser.add_argument('--debug-id', type=int, required=False)

    args = parser.parse_args()

    dataset = datasets.load_dataset("bigcode/bigcodebench")["v0.1.0_hf"]
    dataset = list(iter(dataset))

    if args.debug_id is not None:
        dataset = [d for d in dataset if d["task_id"] == f"BigCodeBench/{args.debug_id}"]
        existing_num_samples = Counter()


    def dataset_iterator() -> Iterator[GenerationInput]:
        for ds_val in dataset:
            prompt = ds_val["complete_prompt"]
            if prompt.endswith("\n"):
                prompt = prompt[:-1]

            prompt = add_imports_from_test(prompt, ds_val["test"])
            yield GenerationInput(task_id=ds_val["task_id"], left_context=prompt, right_context="",
                                  prompt=prompt)


    def output_processor(gen_output: GenerationOutput):
        return {
            "task_id": gen_output.task_id,
            "solution": gen_output.solution
        }


    with create_hallucination_detection_server() as detector:
        sleep(5)
        do_generation_dataset(args=args, ignore_existing=(args.debug_id is not None), dataset=dataset_iterator(),
                              dataset_len=len(dataset), output_processor=output_processor,
                              detector=detector, stopping_criteria=StopWords(stop_words), max_new_tokens=1280)
