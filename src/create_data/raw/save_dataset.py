import os
import json


def save(example):

    output_path = "data/raw"
    os.makedirs(output_path, exist_ok=True)

    with open(
        f"{output_path}/generated.jsonl",
        "a",
        encoding="utf-8"
    ) as f:
        f.write(
            json.dumps(
                example.model_dump(),
                ensure_ascii=False
            ) + "\n"
        )    