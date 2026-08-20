import torch

def response(model, tokenizer, message):
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt = True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad() :

        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]

    generated = output[0][input_len:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

