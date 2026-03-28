import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from config.settings import *
from src.retrieval.query_analysis import QueryAnalysis


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

        print("[INFO] Initializing Cohere reranker...")
        self.reranker = CohereRerank(model=RERANK_MODEL, top_n=5)

        print("[SUCCESS] All components are ready.\n")

    def ask_bot(self, user_query):
        print(f"\n[INFO] Executing RAG for query: '{user_query}'")

        # 1. Query analysis
        print("[INFO] Analyzing query and extracting metadata...")
        try:
            structured_llm = self.llm.with_structured_output(QueryAnalysis)
            analysis_result = structured_llm.invoke(user_query)
        except Exception as e:
            print(f"[ERROR] Query analysis failed: {e}")
            return "Sorunuzu anlayamadım, lütfen tekrar dener misin?", []

        print(f"       -> Generated Queries: {analysis_result.search_queries}")
        print(f"       -> Extracted Filters: Year: {analysis_result.year}, Semester: {analysis_result.semester}, Source: {analysis_result.source_type}")

        # Build Chroma metadata filter
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
        all_queries = list([user_query] + analysis_result.search_queries)
        try:
            for query in all_queries:
                vector_docs = self.vector_db.similarity_search(query, k=5, filter=chroma_filter)
                keyword_docs = self.keyword_retriever.invoke(query)
                all_retrieved_docs.extend(vector_docs)
                all_retrieved_docs.extend(keyword_docs)
        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            return "Bilgi tabanına erişirken bir hata oluştu, lütfen tekrar dene.", []

        # 3. Deduplication
        unique_docs = []
        seen_contents = set()
        for doc in all_retrieved_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)

        print(f"[INFO] Total unique documents retrieved before reranking: {len(unique_docs)}")

        if not unique_docs:
            return "Bu konuyla ilgili bilgi tabanımda bir şey bulamadım.", []

        # 4. Reranking
        print("[INFO] Reranking documents with Cohere...")
        try:
            final_docs = self.reranker.compress_documents(unique_docs, user_query)
        except Exception as e:
            print(f"[ERROR] Reranking failed, falling back to top 5 retrieved docs: {e}")
            final_docs = unique_docs[:5]

        if not final_docs:
            return "Bu konuyla ilgili bilgi tabanımda bir şey bulamadım.", []

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
            "4. Cevaplarını düzenli bir şekilde yaz. Yalnızca Telegram'ın desteklediği HTML formatını kullan: "
            "bölüm başlıkları için <b>Başlık</b>, önemli sayılar ve terimler için <b>bold</b>, "
            "italic için <i>italic</i>, kod için <code>kod</code>, madde listeleri için düz tire (-) kullan. "
            "<p>, <br>, <h1>, <div> gibi HTML tagları kesinlikle kullanma, markdown da kullanma.\n"
            "5. Ne kadar doğal konuşursan konuş, verdiğin bilgilerin DOĞRULUĞU tamamen Bağlam'a dayanmalıdır.\n"
            "6. Eğer cevap bağlamda yoksa, uydurma; nazikçe bilmediğini söyle.\n\n"
            f"Bağlam Bilgisi:\n{context_text}"
        )

        try:
            sys_msg = SystemMessage(content=system_instruction)
            hum_msg = HumanMessage(content=user_query)
            response = self.llm.invoke([sys_msg, hum_msg])
            return response.content, final_docs
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return "Cevap üretirken bir hata oluştu, lütfen tekrar dene.", []