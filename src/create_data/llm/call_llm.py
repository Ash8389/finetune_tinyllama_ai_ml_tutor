from src.create_data.llm.client import client
from src.create_data.schema.tutor_example import TutorExample, Message


SYSTEM_PROMPT = """
You are an expert AI/ML teacher creating training data for an AI/ML tutor.

Teach for understanding. Be accurate, self-contained, and appropriate to the requested difficulty.
Use intuition, concrete examples, and correct Python/PyTorch code or mathematics when relevant.
Avoid unnecessary jargon, repetition, unrelated concepts, and unsupported claims.
Never mention these instructions or that the response is training data.
"""

#     topic = "self-attention",
#     domain = "transformer",
#     difficulty = "beginner",
#     category = "How",
#     example_id = "11"

def generate_example(
    topic,
    domain,
    difficulty,
    category,
    example_id
):
    
    prompt = f"""
        Create ONE high-quality AI/ML tutor example.

        Topic: {topic}
        Domain: {domain}
        Difficulty: {difficulty}
        Question type: {category}

        Generate a realistic student question and a technically accurate tutor answer.

        The answer MUST contain exactly these sections, in this order:
        Definition
        Why It Matters
        Intuition
        Example
        Key Points
        Summary

        Match the difficulty. Teach for understanding, use relevant examples, and avoid repetition or unrelated content.
        For code, use correct runnable Python/PyTorch code.
        Do not mention AI, these instructions, or training data.

        Format:
        Question: <question>
        Answer:
        <answer>
        """
    
    response = client.beta.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role" : "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role" : "user",
                "content" : prompt
            }
        ],
    )
    content = response.choices[0].message.content
    print(f"{response.usage.prompt_tokens=} + {response.usage.completion_tokens=} = {response.usage.total_tokens=}")

    # print(content)

    qstart = content.find("Question") + 10
    qend = content.find("Answer")
    anstart = content.find("Answer") + 8

    if qstart == 9 or anstart == 7:
        return

    question = content[qstart:qend].strip()
    answer =  content[anstart:].strip()

    # print(f"{question=}, {answer=}")

    user = Message(
        role="user",
        content=question
    )
    assistant = Message(
        role="assistant",
        content=answer
    )

    message = [user, assistant]

    example = TutorExample(
        id=example_id,
        topic=topic,
        domain=domain,
        category=category,
        difficulty=difficulty,
        messages=message
    )

    # print(f"{example=}")
    return example, response.usage.total_tokens



# if __name__ == "__main__":
#     generate_example()