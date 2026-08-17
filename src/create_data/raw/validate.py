def validate_example(example):

    if len(example.messages) != 2:
        return False

    if example.messages[0].role != "user":
        return False

    if example.messages[1].role != "assistant":
        return False

    question = example.messages[0].content.strip()
    answer = example.messages[1].content.strip()

    if not question:
        return False

    if not answer:
        return False

    if len(answer) < 100:
        return False

    return True