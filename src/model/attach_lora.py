from peft import get_peft_model

def peft_model_load(base_model, lora_config):
    model = get_peft_model(
        base_model,
        lora_config
    )

    return model