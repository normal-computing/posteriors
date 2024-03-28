from itertools import groupby
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


def load_lora(model_config):
    # Load full language model
    full_model = AutoModelForCausalLM.from_pretrained(
        model_config.pretrained_model_name_or_path
    )

    # Make LoRA
    WEIGHTS_TO_LORA = ["q_proj", "v_proj", "o_proj"]
    modules = list(full_model.model.layers.named_parameters())

    # Get layer index, name for layers to adapt
    module_names_with_layer = [
        (name.split(".")[0], f"layers.{name.strip('.weight')}")
        for name, _ in modules
        if any(
            sub in name
            for sub in ["self_attn.{sub}".format(sub=sub) for sub in WEIGHTS_TO_LORA]
        )
    ]

    # Subset of layers to adapt
    if model_config.lora_config.target_modules == "last_layer":
        modules = [
            [layer for _, layer in list(group)]
            for _, group in groupby(module_names_with_layer, key=lambda x: x[0])
        ][-1]
    else:
        modules = [name for layer, name in module_names_with_layer]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=modules,
        r=model_config.lora_config.r,
        lora_alpha=model_config.lora_config.alpha,
        lora_dropout=model_config.lora_config.dropout,
    )

    model = get_peft_model(full_model, peft_config)
    return model
