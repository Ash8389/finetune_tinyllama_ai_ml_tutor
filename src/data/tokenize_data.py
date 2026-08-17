
def tokenize_data(dataset, tokenizer):
    text = tokenizer.apply_chat_template(
        dataset["message"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text" : text}