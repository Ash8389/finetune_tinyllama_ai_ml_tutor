# from src.create_data.llm.call_llm import generate_example
# from src.create_data.raw.question_type import QUESTION_TYPES
# from src.create_data.raw.topics import topics
# from src.create_data.raw.save_dataset import save
# from src.create_data.raw.validate import validate_example

# import time

# def generate():

#     examples = []

#     for i in range(50):
#         topic_info = topics[i % len(topics)]
#         question_type = QUESTION_TYPES[i % len(QUESTION_TYPES)]

#         example = generate_example(
#             topic=topic_info["topic"],
#             domain=topic_info["domain"],
#             difficulty=topic_info["difficulty"],
#             category=question_type,
#             example_id= f"tutor_{i:04d}"
#         )

#         if len(examples) >= 5:
#             valid_ex = [
#                 ex
#                 for ex in examples
#                 if validate_example(example=example)
#             ]

#             save(valid_ex)

#         examples.append(example)
#         print(f"Generated : {example.id}")
#         time.sleep(2.5)

#     return examples



# # if __name__ == "__main__":
# #     examples = generate()

# #     validated_examples = [
# #         example
# #         for example in examples
# #         if validate_example(example)
# #     ]

# #     for ex in validated_examples:
# #         print(ex.model_dump_json(indent=2))

# #     save(examples=validated_examples)