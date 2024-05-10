#!/bin/bash

device="cuda:1"

# List of temperature params
temperatures=(0.03 0.1 0.3 1.0 3.0)

# Base directories
base_configs=(
    "examples/imdb/results/laplace_fisher"
    "examples/imdb/results/laplace_ggn"
    "examples/imdb/results/vi"
    "examples/imdb/results/sghmc_serial"
    "examples/imdb/results/sghmc_parallel/sghmc_parallel"
)


for temperature in "${temperatures[@]}"; do
    for base_config in "${base_configs[@]}"; do
        config_file="${base_config}_temp${temperature//./-}/config.py"
        echo "Executing $config_file"
        PYTHONPATH=. python examples/imdb/test.py --config "$config_file" \
        --device $device
    done
done
