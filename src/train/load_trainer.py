from trl import SFTTrainer


def train(model, train, validate, tokenizer ,args):
    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=validate,
        processing_class=tokenizer,
        args=args,
    )

    return trainer
