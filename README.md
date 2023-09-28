# bayes-lms

Idea: Use [SGHMC](https://arxiv.org/abs/1506.04696) to train a small Bayesian transformer on top of Llama-2-13b

Problem to work on: https://www.tensorflow.org/text/tutorials/uncertainty_quantification_with_sngp_bert
(text classification problem)


- `clinic-oos.py` downloads and saves all inputs
- `final_hidden_state.py` runs Llama-2-13b on all input and saves final hidden state (1.2GB file). Hopefully we won't have to touch Llama-2-13b again after this.


Steps:

1. Run Llama-2-13b on inputs, store final hidden states. Done ✅
2. Code up small transformer that takes final hidden state as input and outputs classification logits. Done ✅
3. Explore choices of stepsizes with zero friction. Done ✅
4. Train with SGHMC with varying friction parameter. Done ✅
5. Plot results. Done ✅



