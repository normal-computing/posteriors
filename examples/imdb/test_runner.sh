#!/bin/bash

device="cuda:1"

# List of temperature params
temperatures=(0.03 0.1 0.3 1.0 3.0)
seeds=$(seq 1 5)

# Base directories
base_configs=(
    # "examples/imdb/results/vi"
    # "examples/imdb/results/sghmc_serial"
    "examples/imdb/results/sghmc_parallel"
)

for temperature in "${temperatures[@]}"; do
    for seed in $seeds; do
        for base_config in "${base_configs[@]}"; do
            config_file="${base_config}_seed${seed}_temp${temperature//./-}/config.py"
            echo "Executing $config_file"
            PYTHONPATH=. python examples/imdb/test.py --config "$config_file" \
            --device $device
        done
    done
done



# # Laplace

# temperatures=(0.03 0.1 0.3 1.0 3.0)
# seeds=$(seq 2)


# # Base directories
# base_configs=(
#     "examples/imdb/results/laplace_fisher"
#     # "examples/imdb/results/laplace_ggn"
# )

# for temperature in "${temperatures[@]}"; do
#     for seed in $seeds; do
#         for base_config in "${base_configs[@]}"; do
#             config_file="${base_config}_seed${seed}/config.py"
#             echo "Executing $config_file"
#             PYTHONPATH=. python examples/imdb/test.py --config "$config_file" \
#             --device $device --temperature $temperature
#         done
#     done
# done


# # MLE and MAP
# # List of temperature params
# seeds=$(seq 1 5)

# # Base directories
# base_configs=(
#     "examples/imdb/results/mle"
#     "examples/imdb/results/map"
# )

# for seed in $seeds; do
#     for base_config in "${base_configs[@]}"; do
#         config_file="${base_config}_seed${seed}/config.py"
#         echo "Executing $config_file"
#         PYTHONPATH=. python examples/imdb/test.py --config "$config_file" \
#         --device $device
#     done
# done