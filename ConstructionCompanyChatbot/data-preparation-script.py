import os
import json
import random
from openai import OpenAI

# Set your OpenAI API key
api_key = "sk-<YOUR_API_KEY>"
client = OpenAI(api_key=api_key)


def generate_construction_data(num_entries=25):
    topics = [
        "safety", "policy", "project", "equipment", "general",
        "regulations", "sustainability", "training", "quality control",
        "client relations"
    ]

    data = []

    for i in range(num_entries):
        topic = random.choice(topics)
        prompt = f"Generate a brief, informative paragraph about {topic} in the context of a construction company. The content should be specific and practical."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant for a construction company, providing concise and informative content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()

        data.append({
            "id": f"{topic}_{i+1}",
            "category": topic,
            "content": content
        })

    return data


def save_data(data, output_dir="construction_company_data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save as individual text files
    for item in data:
        filename = f"{item['id']}.txt"
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(item['content'])

    # Save as a single JSON file
    json_filename = "construction_data.json"
    with open(os.path.join(output_dir, json_filename), "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    generated_data = generate_construction_data()
    save_data(generated_data)
    print(
        f"Generated {len(generated_data)} data entries in 'construction_company_data' directory.")
