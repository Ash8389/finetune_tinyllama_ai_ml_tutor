from src.inference.load_models import load
from src.inference.config_bnb import bnb_config
from src.inference.load_base_model import load_base_model

from transformers import AutoTokenizer

def pipeline():
    adapter_path = "ash270/tinyllama-ai-ml-tutor-qlora"
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    config = bnb_config()

    base_model = load_base_model(base_model_name, config)
    
    fine_tuned_base = load_base_model(base_model_name, config)
    fine_tuned_model = load(fine_tuned_base, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return base_model, fine_tuned_model, tokenizer