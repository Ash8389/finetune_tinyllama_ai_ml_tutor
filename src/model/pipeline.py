from src.model.config_bnb import bnb_config
from src.model.load_base_model import load_base_model
from src.model.config_lora import config_lora
from src.model.attach_lora import peft_model_load

from peft import prepare_model_for_kbit_training

def pipeline(model_name: str):
    bnb = bnb_config()
    base_model = load_base_model(model_name=model_name, bnb_config=bnb)

    training_model = prepare_model_for_kbit_training(base_model)

    lora_config=config_lora()
    training_model = peft_model_load(base_model=training_model, lora_config=lora_config)

    return base_model, training_model
