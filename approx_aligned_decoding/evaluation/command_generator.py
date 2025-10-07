from itertools import product


# For some reason the bigcodebench evaluation script completely breaks when run in a .sh script
# (Or in docker, though that's a different issue...)
# As a hack, this script makes some nice copy-pasteable lines
# Evaluation is separate because there are different parallelism limits

# These are imported in command_generator

models = [
    "bigcode/starcoder2-15b",
    "bigcode/starcoder2-7b"
]

strategies = [
    ("unconstrained", ""),
    ("fastgad", "--use-detector --backtrack-strategy fastgad --backtrack-limit 1000"),
    ("gad", "--use-detector --backtrack-strategy gad --backtrack-limit 1000"),
    ("naive", "--use-detector --backtrack-strategy naive --move-hallu-idx-forward --backtrack-limit 1000"),
]


def get_base_name(model, strategy_name):
    return f"evaluation/{model.replace('/', '-')}-{strategy_name}"


def generate():
    port_num = 5003
    gpu_num = 0

    for model, (strategy_name, strategy_args) in product(models, strategies):
        print(f"# {model} {strategy_name}")
        base_name = get_base_name(model, strategy_name)
        print(
            "conda activate approx-aligned-decoding && " +
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PORT={port_num} PYTHONPATH=. " +
            "python approx_aligned_decoding/evaluation/generate_bigcodebench.py" +
            f" --model {model} --temperature 0.8 --top-p 0.95 --num-samples 5 --device-model \"cuda:{gpu_num % 8}\""
            + f" {strategy_args} --output {base_name}.jsonl"
        )
        print(f"conda activate bigcodebench && bigcodebench.sanitize {base_name}.jsonl && " +
              f"bigcodebench.evaluate --subset complete --parallel 16 --samples {base_name}-sanitized.jsonl")
        print("\n")

        port_num += 1
        gpu_num += 1


if __name__ == '__main__':
    generate()
