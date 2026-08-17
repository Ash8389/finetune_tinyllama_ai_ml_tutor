from src.train.training_argument import training_args
from src.train.load_trainer import train
from src.train.save_adapter import save

def pipeline(dataset, model, tokenizer):
    train_data = dataset["train"]
    validate = dataset["validate"]

    args = training_args()
    trainer = train(model=model, train=train_data, validate=validate,tokenizer=tokenizer, args=args)
    
    trainer.train()
    save(trainer=trainer, tokenizer=tokenizer)

    print("### TRAINING COOMPLETED ###")