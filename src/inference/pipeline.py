from src.inference.load_models import load
from src.inference.generate_response import response
from src.inference.config_bnb import bnb_config
from src.inference.load_base_model import load_base_model

from transformers import AutoTokenizer

def pipeline(base_model_name, messages):
    adapter_path = "outputs/final_adapter"

    config = bnb_config()

    base_model = load_base_model(base_model_name, config)
    fine_tuned_base = load_base_model(base_model_name, config)
    fine_tuned_model = load(fine_tuned_base, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)


    for i in range(5):
        question = [messages[i]["messages"][0]]
        finetuned_response = response(question ,fine_tuned_model, tokenizer)
        basemodel_response = response(question ,base_model, tokenizer)
        print(f"{question[0]['content']=}")
        print(f"{finetuned_response=}")
        print(f"{basemodel_response=}\n")

    return {finetuned_response, basemodel_response}