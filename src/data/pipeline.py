from src.data.load_data_set import load
from src.data.to_chat_format import chat_format
from src.data.tokenize_data import tokenize_data
from src.data.clean_dataset import clean

def pipeline(files_path):

    datasets = load(dataset_paths=files_path)

    # chat_format_dataset = base_dataset.map(chat_format)

    # tokenized_dataset = chat_format_dataset.map(
    #     lambda x: tokenize_data(x, tokenizer = tokenizer)
    # )
    train = datasets["train"]
    validate = datasets["validate"]
    test = datasets["test"]

    train = clean(dataset = train)
    validate = clean(dataset = validate)
    test = clean(dataset = test)

    return train, validate, test