import os
import json
import random
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from config.settings import *

def generate_test_set(output_path=TEST_QUESTIONS_PATH):
    print("[INFO] Initializing Gemini models...")
    
    base_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        response_mime_type="application/json" 
    )
    
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, 
        google_api_key=GEMINI_API_KEY
    )

    print("[INFO] Connecting to Chroma DB and extracting chunks...")
    vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=base_embeddings)
    data = vector_db.get()
    
    documents = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    print(f"[INFO] Loaded {len(documents)} total chunks from DB.")

    print("[INFO] Stratifying and sampling documents...")
    official_stats = []
    official_others = []
    opinions = []

    for doc in documents:
        source = doc.metadata.get("source_type", "")
        doc_type = doc.metadata.get("doc_type", "")

        if source == "official":
            if doc_type == "statistics":
                official_stats.append(doc)
            else:
                official_others.append(doc)
        elif source == "opinion":
            opinions.append(doc)

    random.seed(42) 
    random.shuffle(official_stats)
    random.shuffle(official_others)
    random.shuffle(opinions)

    sampled_docs = official_stats[:20] + official_others[:40] + opinions[:40]
    random.shuffle(sampled_docs)
    print(f"[INFO] Sampled {len(sampled_docs)} chunks successfully.")

    print("[INFO] Setting up Ragas Testset Generator...")
    generator = TestsetGenerator.from_langchain(
        llm=base_llm,
        embedding_model=base_embeddings,
        llm_context=(
            "You are a helpful assistant generating test data. "
            "CRITICAL: All generated content (questions, answers) MUST be in Turkish. "
            "However, keep the JSON keys strictly in English (e.g., 'summary', 'question')."
        )
    )

    print("[INFO] Generating test dataset from CHUNKS...")
    testset = generator.generate_with_chunks(
        chunks=sampled_docs,
        testset_size=20 
    )

    print("[INFO] Exporting to JSON...")
    df = testset.to_pandas()
    df = df.fillna("")
    records = df.to_dict(orient="records")
    
    for record in records:
        if "contexts" in record:
            record["contexts"] = [str(c) for c in record["contexts"]]
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] {len(records)} test questions saved to '{output_path}'.")
    return records


if __name__ == "__main__":
    generate_test_set()