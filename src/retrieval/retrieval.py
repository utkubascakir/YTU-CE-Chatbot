import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from config.settings import *


class QueryAnalysis(BaseModel):
    """Analyzes the user query to extract metadata filters and generate search variations."""
    search_queries: List[str] = Field(
        description="Generate 3 different search query variations of the original query to improve retrieval recall."
    )
    year: Optional[int] = Field(
        None, description="If the user mentions a specific year (1, 2, 3, or 4), extract it as an integer."
    )
    semester: Optional[str] = Field(
        None, description="If the user mentions a semester ('Güz' or 'Bahar'), extract it."
    )
    source_type: Optional[str] = Field(
        None, description="If the user asks for official rules/regulations, output 'official'. If they ask for advice, opinions, or comments, output 'opinion'."
    )


class YTUCEAssistant:
    def __init__(self, db_path=CHROMA_DB_PATH):
        print("[INFO] Initializing embedding model...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY
        )
        
        print("[INFO] Connecting to Chroma vector database...")
        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        
        # Initialize BM25 with all documents in Chroma
        print("[INFO] Setting up BM25 keyword retriever...")
        data = self.vector_db.get()
        extracted_docs = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(data['documents'], data['metadatas'])
        ]
        self.keyword_retriever = BM25Retriever.from_documents(extracted_docs)
        self.keyword_retriever.k = 5 
        
        print("[INFO] Initializing Gemini LLM...")
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2
        )
        
        print("[SUCCESS] All components are ready.\n")

    # Core RAG execution
    def ask_bot(self, user_query):
        print(f"\n[INFO] Executing RAG for query: '{user_query}'")
        
        # 1. Query analysis
        print("[INFO] Analyzing query and extracting metadata...")
        structured_llm = self.llm.with_structured_output(QueryAnalysis)
        analysis_result = structured_llm.invoke(user_query)
        
        print(f"       -> Generated Queries: {analysis_result.search_queries}")
        print(f"       -> Extracted Filters: Year: {analysis_result.year}, Semester: {analysis_result.semester}, Source: {analysis_result.source_type}")

        # Chroma filter
        chroma_filter = {}
        filter_list = []
        
        if analysis_result.year:
            filter_list.append({"year": {"$eq": analysis_result.year}})
        if analysis_result.semester:
            filter_list.append({"semester": {"$eq": analysis_result.semester}})
        if analysis_result.source_type:
            filter_list.append({"source_type": {"$eq": analysis_result.source_type}})
            
        if len(filter_list) > 1:
            chroma_filter = {"$and": filter_list}
        elif len(filter_list) == 1:
            chroma_filter = filter_list[0]
        else:
            chroma_filter = None

        # 2. Hybrid retrieval
        print("[INFO] Retrieving documents using Multi-Query Hybrid Search...")
        all_retrieved_docs = []
        
        for query in analysis_result.search_queries:
            vector_docs = self.vector_db.similarity_search(query, k=5, filter=chroma_filter)
            keyword_docs = self.keyword_retriever.invoke(query)
            
            all_retrieved_docs.extend(vector_docs)
            all_retrieved_docs.extend(keyword_docs)

        # 3. Deduplication
        unique_docs = []
        seen_contents = set()
        for doc in all_retrieved_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
                
        print(f"[INFO] Total unique documents retrieved before reranking: {len(unique_docs)}")

        # 4. Reranking
        if not unique_docs:
            return "I couldn't find any relevant information regarding your question.", []
            
        print("[INFO] Reranking documents with Cohere...")
        reranker = CohereRerank(model=RERANK_MODEL, top_n=5) 
        final_docs = reranker.compress_documents(unique_docs, user_query)

        # 5. Prepare context
        context_parts = []
        for doc in final_docs:
            if doc.metadata.get("doc_type") == "statistics" and "original_table" in doc.metadata:
                context_parts.append(doc.metadata["original_table"])
            else:
                context_parts.append(doc.page_content)
                
        context_text = "\n\n---\n\n".join(context_parts)
        
        # 6. Generation
        print("[INFO] Generating final response...")
        system_instruction = (
            "Sen bir üniversite öğrencisine yardım eden, oldukça cana yakın, profesyonel ve anlayışlı bir akademik danışmansın. "
            "Görevin, aşağıda sana verilen resmi yönetmelik metinlerini, tabloları ve öğrenci yorumlarını (Bağlam) analiz etmek ve "
            "öğrencinin sorusunu resmi kurum dilinden arındırarak, günlük ve doğal bir konuşma diliyle cevaplamaktır.\n\n"
            
            "KURALLAR:\n"
            "1. Asla belgelerdeki cümleleri birebir kopyala-yapıştır yapma. Bilgiyi özümse ve kendi cümlelerinle anlat. "
            "Öğrenciye 'yapmalısın', 'etmelisin' gibi doğrudan hitap et.\n"
            "2. Eğer bağlamda bir MAAŞ TABLOSU veya İSTATİSTİK varsa, sayıları doğru oku ve karşılaştırmalı olarak güzelce özetle.\n"
            "3. Eğer bağlamda ÖĞRENCİ YORUMLARI (opinion) varsa, bunları 'Öğrencilerin genel görüşüne göre...' şeklinde belirt.\n"
            "4. Cevaplarını maddeler halinde düzenle ve **önemli sayıları, ortalamaları veya tarihleri** kalın yazarak vurgula.\n"
            "5. Ne kadar doğal konuşursan konuş, verdiğin bilgilerin DOĞRULUĞU tamamen Bağlam'a dayanmalıdır.\n"
            "6. Eğer cevap bağlamda yoksa, uydurma; nazikçe bilmediğini söyle.\n\n"
            f"Bağlam Bilgisi:\n{context_text}" 
        )
        
        sys_msg = SystemMessage(content=system_instruction)
        hum_msg = HumanMessage(content=user_query)
        
        response = self.llm.invoke([sys_msg, hum_msg])
        
        return response.content, final_docs


if __name__ == "__main__":
    # Test 
    bot = YTUCEAssistant()
    
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        answer, sources = bot.ask_bot(query)
        
        print("\n" + "="*50)
        print("[ANSWER]")
        print("="*50)
        print(answer)
        
        print("\n" + "-"*50)
        print("[SOURCES]")
        print("-"*50)
        for idx, doc in enumerate(sources):
            source_type = doc.metadata.get('source_type', 'Unknown')
            doc_type = doc.metadata.get('doc_type', 'Unknown')
            print(f"{idx+1}. Type: {source_type}/{doc_type} | Content: {doc.page_content[:150]}...")