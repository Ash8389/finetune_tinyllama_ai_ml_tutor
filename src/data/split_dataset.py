import json
import os
import random

SEED = 40

INPUT_FILE = "data/raw/generated.jsonl"
TRAIN_FILE = "data/raw/train.jsonl"
VALIDATE_FILE = "data/raw/validate.jsonl"
TEST_FILE = "data/raw/test.jsonl"

def read_data(file_path):

    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if(line.strip()):
                examples.append(json.loads(line))

    return examples


def write_data(examples, file_path):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    with open(file_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                ) + "\n"
            )

def split_dataset(
        examples,
        train_data_ratio = 0.8,
        validate_data_ratio = 0.1,
        test_data_ratio = 0.1

):

    random.seed(SEED)
    random.shuffle(examples)

    total = int(len(examples))

    train_end = int(total * train_data_ratio)
    validate_end = (train_end + int(total * validate_data_ratio))

    train_data = examples[:train_end]
    validate_data = examples[train_end:validate_end]
    test_data = examples[validate_end:]

    return train_data, validate_data, test_data

def main():

    examples = read_data(INPUT_FILE)

    train_data, validate_data, test_data = split_dataset(examples)

    write_data(train_data, TRAIN_FILE)
    write_data(validate_data, VALIDATE_FILE)
    write_data(test_data, TEST_FILE)

    print(f"{len(examples)=}")
    print(f"{len(train_data)=}")
    print(f"{len(validate_data)=}")
    print(f"{len(test_data)=}")


if __name__ == "__main__":
    main()