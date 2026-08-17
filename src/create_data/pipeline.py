from src.create_data.llm.call_llm import generate_example
from src.create_data.raw.question_type import QUESTION_TYPES
from src.create_data.raw.topics import topics
from src.create_data.raw.save_dataset import save
from src.create_data.raw.validate import validate_example

import time
from openai import RateLimitError

def pipeline():

    total_token_used = 0
    for i in range(2261, 2300):
        topic_info = topics[i % len(topics)]
        question_type = QUESTION_TYPES[i % len(QUESTION_TYPES)]

        try:

            example, token_used = generate_example(
                topic=topic_info["topic"],
                domain=topic_info["domain"],
                difficulty=topic_info["difficulty"],
                category=question_type,
                example_id= f"tutor_{i:04d}"
            )
            total_token_used += token_used
            print(total_token_used)

            if validate_example(example=example):
                save(example)
                print(f"Saved Example : {example.id}")

            else:
                print(f"Invalid Example : {example.id}")
        

        except RateLimitError as re:
            print(f"Rate Limit Error : {re}")
            break
        except Exception as e:
            print(e)

        time.sleep(3)

if __name__ == "__main__":
    pipeline()