# Bayesian Ensemble Language Model #

We create an ensemble last layer on top of Llama3 to perform uncertainty quantification. We fine-tune the last attention layer in the model and achieve a distribution over distributions by grabbing 10 copies of the weights over 10 different training trajectories.

## Installation ##

1. We need to install TQA dataset (https://allenai.org/data/tqa):

    ```wget https://ai2-public-datasets.s3.amazonaws.com/tqa/tqa_train_val_test.zip && unzip -q tqa_train_val_test.zip```

2. Download the HuggingFace Llama3 weights here: https://huggingface.co/meta-llama/Meta-Llama-3-8B.

## Running the Code ##

### _Training Instructions_ ###

Make sure you change the used config to match the correct paths (i.e., you may need to change the dataset path value).

For training, run the following:

```python train_ensemble.py --base configs/training/ensemble_bayes.yaml --devices 0,```

If you have additional GPUs you can add them by specifying `0,1,2,3,...` after the `--devices` flag.


### _Evaluation Instructions_ ###

Ensure that your config is pointing to a path with all the ensemble weights you would like to load in `checkpoints_folder`.

To evaluate, run the following:

```python run_eval.py --base configs/evaluation/eval_bayes_ensemble.yaml --output ensemble.pkl```

To recreate plots and get metrics, run ```python plot.py```.

## Details ##

The training code does not need a separate Llama3 model code. It can work out of the box. But, for inference, we made modifications you can find in `llama3/bayesllama.py`

The statements referred to in the paper are found in `llama3/data/statements.py`


