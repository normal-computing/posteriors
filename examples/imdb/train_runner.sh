#!/bin/bash

device="cuda:0"

# temperatures=(0.03 0.1 0.3 1.0 3.0)
# config="examples/imdb/configs/vi.py"
# epochs=30
# seeds=$(seq 1 5)

# temperatures=(0.03 0.1 0.3 1.0 3.0)
# config="examples/imdb/configs/sghmc_serial.py"
# epochs=60
# seeds=$(seq 1 5)

temperatures=(0.03 0.1 0.3 1.0 3.0)
config="examples/imdb/configs/sghmc_parallel.py"
epochs=30
seeds=$(seq 1 35)

# temperatures=(1.0)
# config="examples/imdb/configs/map.py"
# epochs=30
# seeds=$(seq 1 5)

# temperatures=(1.0)
# config="examples/imdb/configs/mle.py"
# epochs=30
# seeds=$(seq 1 5)

# temperatures=(1.0)
# config="examples/imdb/configs/laplace_fisher.py"
# epochs=1
# seeds=$(seq 1 5)

# temperatures=(1.0)
# config="examples/imdb/configs/laplace_ggn.py"
# epochs=1
# seeds=$(seq 1 5)

for seed in $seeds
do
    for temp in "${temperatures[@]}"
    do
        PYTHONPATH=. python examples/imdb/train.py --config $config --device $device \
        --temperature $temp --epochs $epochs --seed $seed
    done
done