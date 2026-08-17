from peft import PeftModel

def load(base_model, adapter_path):
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )

    return model