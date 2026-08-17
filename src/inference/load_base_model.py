from transformers import AutoModelForCausalLM

def load_base_model(model_name: str, bnb_config):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return base_model