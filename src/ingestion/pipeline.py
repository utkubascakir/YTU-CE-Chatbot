import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.ingestion.process_jsons import process_courses_json, process_semesters_json, process_instructors_json
from src.ingestion.process_markdown import process_markdown_tables
from config.settings import *


def process_pdfs(data_path):
    if not os.path.exists(data_path):
        return []
    
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # PDFs are official documents, adding metadata for model understanding
    for chunk in chunks:
        chunk.metadata["source_type"] = "official"
        chunk.metadata["doc_type"] = "regulation"
        
    print(f"[INFO] Number of processed PDF chunks: {len(chunks)}")
    return chunks


def process_jsons(json_folder_path):
    if not os.path.exists(json_folder_path):
        print(f"[WARNING] JSON folder '{json_folder_path}' not found.")
        return []
        
    all_docs = []
    
    json_files = {
        "dersler.json": process_courses_json,
        "donemler.json": process_semesters_json,
        "hocalar.json": process_instructors_json
    }
    
    for file_name, process_func in json_files.items():
        full_path = os.path.join(json_folder_path, file_name)
        
        if os.path.exists(full_path):
            try:
                docs = process_func(full_path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"[ERROR] Failed to process {file_name}: {e}")
        else:
            print(f"[WARNING] File not found: {full_path}")
            
    return all_docs


def create_unified_vector_db(all_documents, persist_directory=CHROMA_DB_PATH):
    print("[INFO] Initializing Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )
    print(f"[INFO] Embedding a total of {len(all_documents)} chunks into Chroma database...")
    vector_store = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"[SUCCESS] Vector store created and persisted at: '{persist_directory}'.\n")
    return vector_store


def main():
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY not found.")
        exit(1)
        
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0)
        
    print("[INFO] Starting ingestion pipeline...\n")
    yonerge_path = os.path.join(DATA_PATH, "yonerge_pdf")
    comments_path = os.path.join(DATA_PATH, "student_comments")
    statistics_path = os.path.join(DATA_PATH, "maas_istatistikleri.md")
    
    pdf_chunks = process_pdfs(yonerge_path)
    json_chunks = process_jsons(comments_path)
    table_chunks = process_markdown_tables(llm=llm, md_path=statistics_path)
    
    all_documents = pdf_chunks + json_chunks + table_chunks
    
    if all_documents:
        create_unified_vector_db(all_documents)
        print("[INFO] Ingestion pipeline completed successfully.")
    else:
        print("[WARNING] No documents found to ingest!")
        
        
if __name__ == "__main__":
    main()