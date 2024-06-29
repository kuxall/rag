import os
import random


def generate_sample_data():
    if not os.path.exists("construction_company_data"):
        os.makedirs("construction_company_data")

    topics = [
        ("safety", "Safety guidelines for working at heights include using proper fall protection equipment, ensuring stable work platforms, and regular safety training."),
        ("policy", "Company policy requires all employees to wear appropriate personal protective equipment (PPE) on construction sites at all times."),
        ("project", "The new office building project is scheduled to begin next month, with an estimated completion time of 18 months."),
        ("equipment", "Heavy machinery operators must perform daily inspections of their equipment before use to ensure safe operation."),
        ("general", "Our company is committed to sustainable construction practices, including the use of eco-friendly materials and energy-efficient designs.")
    ]

    for i in range(10):
        topic, content = random.choice(topics)
        filename = f"{topic}_{i+1}.txt"
        with open(os.path.join("construction_company_data", filename), "w") as f:
            f.write(content)


if __name__ == "__main__":
    generate_sample_data()
    print("Sample data generated in 'construction_company_data' directory.")
