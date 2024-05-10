#!/bin/bash

# List of temperature params
temperatures=(0.03 0.1 0.3 1.0 3.0)


device="cuda:1"


# config="examples/imdb/configs/vi.py"
# epochs=30
# seeds=$(42)

# config="examples/imdb/configs/sghmc_serial.py"
# epochs=60
# seeds=$(42)

config="examples/imdb/configs/sghmc_parallel.py"
epochs=30
seeds=$(seq 2 19)

for seed in $seeds
do
    for temp in "${temperatures[@]}"
    do
        PYTHONPATH=. python examples/imdb/train.py --config $config --device $device \
        --temperature $temp --epochs $epochs --seed $seed
    done
done