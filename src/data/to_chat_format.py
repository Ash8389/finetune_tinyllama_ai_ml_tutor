

def chat_format(dataset):
    user_message = dataset['instruction']

    if dataset['context']:
        user_message += "\n\nContext : \n" + dataset['context']

    message = [
        {
            "role" : "user",
            "content" : user_message 
        },
        {
            "role" : "assistant",
            "content" : dataset['response']
        }
    ]

    return {"message" : message}