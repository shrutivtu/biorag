# main.py

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

# main.py (inside load_data function)
def load_data():
    print("Loading datasets...")
    qa_data = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split="test")
    corpus_data = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")

    # Build a lookup from corpus ID to passage text
    id_to_text = {int(doc["id"]): doc["passage"] for doc in corpus_data}

    questions, answers, passages = [], [], []

    for item in qa_data:
        questions.append(item["question"])
        answers.append(item["answer"])
        passage_ids = [int(pid) for pid in item["relevant_passage_ids"].strip("[]").split(",")]
        related_passages = [id_to_text.get(pid, "") for pid in passage_ids]
        passages.append(" ".join(related_passages))  # Combine all relevant passages into one string

    return passages, questions, answers



def build_faiss_index(passages, encoder):
    print("Encoding passages...")
    embeddings = encoder.encode(passages, show_progress_bar=True, convert_to_numpy=True)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def generate_answer(query, encoder, index, passages, tokenizer, generator, top_k=3):
    print(f"Retrieving for query: {query}")
    query_embedding = encoder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved = [passages[i] for i in indices[0]]
    context = " ".join(retrieved)

    input_text = f"question: {query} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    output = generator.generate(inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    # Load everything
    passages, questions, answers = load_data()

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    generator = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

    index, _ = build_faiss_index(passages, encoder)

    # Ask user for a query
    while True:
        query = input("\nEnter your biomedical question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        answer = generate_answer(query, encoder, index, passages, tokenizer, generator)
        print(f"\nGenerated Answer:\n{answer}")

if __name__ == "__main__":
    main()
