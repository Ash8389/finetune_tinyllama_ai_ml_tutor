def save(trainer, tokenizer):
    tokenizer.save_pretrained("./outputs/final_adapter")
    trainer.save_model("./outputs/final_adapter")
