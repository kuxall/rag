import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

def load_resources():
    # Load the FAISS index
    index = faiss.read_index("construction_faiss_index")
    
    # Load the id mapping
    with open('id_mapping.json', 'r') as f:
        id_mapping = json.load(f)
    
    # Load the original data
    with open('D:/rag/ConstructionCompanyChatbot/construction_company_data/construction_data.json', 'r') as f:
        data = json.load(f)
    
    # Create a mapping from id to data
    id_to_data = {item['id']: item for item in data}
    
    return index, id_mapping, id_to_data

def query_and_retrieve(query: str, top_k: int = 3) -> List[Dict]:
    index, id_mapping, id_to_data = load_resources()
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embedding for the query
    query_embedding = model.encode([query])[0]
    
    # Perform the search
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    # Retrieve the corresponding documents
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        doc_id = id_mapping[str(idx)]
        doc_data = id_to_data[doc_id]
        results.append({
            "content": doc_data['content'],
            "category": doc_data['category'],
            "relevance_score": 1 / (1 + distance)  # Convert distance to a relevance score
        })
    
    return results

# Example usage
if __name__ == "__main__":
    query = "What are the safety guidelines for working at heights?"
    results = query_and_retrieve(query)
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Category: {result['category']}")
        print(f"Relevance Score: {result['relevance_score']:.4f}")
        print(f"Content: {result['content'][:200]}...\n")
