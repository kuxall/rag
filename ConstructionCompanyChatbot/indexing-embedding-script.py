import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_processed_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def create_embeddings(model, texts):
    return model.encode(texts)


def create_faiss_index(embeddings, dimension):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)  # type: ignore
    return index


def main():
    # Load the processed data
    data = load_processed_data('D:/rag/ConstructionCompanyChatbot/construction_company_data/construction_data.json')
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings
    texts = [item['content'] for item in data]
    embeddings = create_embeddings(model, texts)

    # Create FAISS index
    dimension = embeddings.shape[1]  # type: ignore
    index = create_faiss_index(embeddings, dimension)

    # Save the index
    faiss.write_index(index, "construction_faiss_index")

    # Save the mapping of index to document id
    id_mapping = {i: item['id'] for i, item in enumerate(data)}
    with open('id_mapping.json', 'w') as f:
        json.dump(id_mapping, f)

    print("Indexing and embedding complete.")


if __name__ == "__main__":
    main()
