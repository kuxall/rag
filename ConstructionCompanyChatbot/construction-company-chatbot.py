from openai import OpenAI
from queryretrievalfunction import query_and_retrieve

# Set your OpenAI API key
api_keys = "sk-<YOUR_API_KEY>"
client = OpenAI(
    # This is the default and can be omitted
    api_key=api_keys,
)


def generate_response(query, context):
    prompt = f"""You are an AI assistant for a construction company. Use the following information to answer the user's question. If the information provided doesn't answer the question directly, use it to infer a helpful response. If you can't answer the question based on the given information, say so politely and offer to help with related topics.

Context:
{context}

User Question: {query}

AI Assistant:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
                "content": "You are an AI assistant for a construction company."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].message.content


def chatbot():
    print("Construction Company Chatbot: Hello! How can I assist you with your construction-related questions?")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print(
                "Construction Company Chatbot: Thank you for using our service. Have a great day!")
            break

        # Retrieve relevant information
        results = query_and_retrieve(user_input)

        # Prepare context for the language model
        context = "\n\n".join(
            [f"Category: {r['category']}\nContent: {r['content']}" for r in results])

        # Generate response
        response = generate_response(user_input, context)

        print("Construction Company Chatbot:", response)


if __name__ == "__main__":
    chatbot()
