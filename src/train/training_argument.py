from trl import SFTConfig

def training_args():
    args = SFTConfig(
        output_dir="./outputs",

        num_train_epochs=3,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=2e-4,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=2,

        lr_scheduler_type="cosine",

        warmup_ratio=0.03,

        logging_steps=10,

        report_to="none",

        fp16=True,

        optim="paged_adamw_8bit",

        seed=42,
    )

    return args