from src.inference.generate_response import response
from src.api.schemas import LlmResponse

class Service:

    def __init__(self, base_model, finetuned_model, tokenizer):
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.tokenizer = tokenizer

    def query_service(self, question):
        basemodel_res = response(self.base_model, self.tokenizer, message=question)
        finetuned_res = response(self.finetuned_model, self.tokenizer, message=question)

        llm_response = LlmResponse(
            finetuned_answer =finetuned_res,
            basemodel_answer = basemodel_res
        )

        return llm_response