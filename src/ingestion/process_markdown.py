import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from config.settings import *


def process_markdown_tables(llm, md_path):
    if not os.path.exists(md_path):
        print(f"[WARNING] Markdown file '{md_path}' not found.")
        return []
        
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 1. Make years and periods more explicit by converting details/summary to headers
    md_text = re.sub(r"<details>\s*<summary>(.*?)</summary>", r"# \1", md_text, flags=re.IGNORECASE)
    md_text = re.sub(r"</details>", "", md_text, flags=re.IGNORECASE)

    # 2. Defining the hierarchy
    headers_to_split_on = [
        ("#", "year_period"),       
        ("###", "employment_type"),  
        ("#####", "category")        
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = markdown_splitter.split_text(md_text)
    
    documents = []
    
    print(f"[INFO] Markdown splitting complete. Found {len(splits)} sections. Summarizing tables with LLM...")
    
    # 3. Process each split
    for split in splits:
        content = split.page_content.strip()
        
        if not content or "|" not in content:
            continue
            
        year_period = split.metadata.get("year_period", "Genel")
        employment_type = split.metadata.get("employment_type", "Genel")
        category = split.metadata.get("category", "Genel")
        
        context = f"Dönem: {year_period} | Çalışma Tipi: {employment_type} | Kategori: {category}"
        
        prompt = (
            f"Sen bir RAG sistemi için teknik döküman yazıyorsun. "
            f"ÇIKTI KURALLARI: Yanıtın doğrudan [BÖLÜM 1] ile başlamalıdır. "
            f"Herhangi bir giriş ifadesi kullanma. "
            f"Aşağıdaki maaş istatistiği tablosunu 3 bölümde özetle.\n\n"
            f"BAĞLAM: {context}\n\n"

            f"[BÖLÜM 1 - Kapsam ve Terminoloji]\n"
            f"TAM OLARAK 3 cümle yaz, fazlası değil:\n"
            f"1. cümle: dönem ({year_period}), çalışma tipi ({employment_type}), kategori ({category}) bilgilerini içersin.\n"
            f"2. cümle: tablonun genel olarak ne anlattığını açıkla (hangi tür verileri, hangi amaçla sunduğunu).\n"
            f"3. cümle: tabloda geçen pozisyon/alan adlarını şu formatta listele: "
            f"'Pozisyon1 (eşanlamlı1, eşanlamlı2), Pozisyon2 (eşanlamlı1, eşanlamlı2), ...' — "
            f"her pozisyon için SADECE 2-3 kelimelik kısa Türkçe/İngilizce eşanlamlı yaz, "
            f"tanım veya açıklama yazma, madde madde listeleme.\n\n"

            f"[BÖLÜM 2 - Sayısal Özet]\n"
            f"Tablodaki HER satırı eksiksiz şu formatta yaz: "
            f"'[Kategori adı]: [ilk yıl] X TL, [ikinci yıl] Y TL, artış oranı %Z.' "
            f"Hiçbir satırı atlama. Sayıları olduğu gibi kullan, yorum veya tahmin ekleme.\n\n"

            f"[BÖLÜM 3 - Anahtar Kelimeler]\n"
            f"SADECE bu tabloya özgü anahtar kelimeleri virgülle ayrılmış düz liste olarak yaz, cümle kurma. "
            f"Şunları mutlaka dahil et: pozisyon/alan adları, yıl ve dönem bilgisi, çalışma tipi, "
            f"varsa şirket adları, varsa tecrübe aralıkları, ve şu sabit kelimeler: "
            f"maaş, ücret, kazanç, gelir, ortalama, artış oranı, istatistik, YTÜ, bilgisayar mühendisliği.\n\n"

            f"Tablo Verisi:\n{content}\n\n"
            f"ÖNEMLİ: Tabloda olmayan hiçbir sayısal veri ekleme."
        )
        
        try:
            summary = llm.invoke(prompt).content.strip()
            
            table_doc = Document(
                page_content=summary, 
                metadata={
                    "source_type": "official",
                    "doc_type": "statistics",
                    "year_period": year_period,
                    "employment_type": employment_type,
                    "category": category,
                    "original_table": content 
                }
            )
            documents.append(table_doc)
            
        except Exception as e:
            print(f"[ERROR] Failed to summarize table in {context}. Error: {e}")
            
    print(f"[INFO] Total of {len(documents)} table chunks summarized and created.")
    return documents


if __name__ == "__main__":
    # Test
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY)
    md_file = os.path.join(DATA_PATH, "maas_istatistikleri.md")
    docs = process_markdown_tables(llm, md_file)
    for doc in docs:
        print(f"\n[SUMMARY] {doc.metadata['year_period']} | {doc.metadata['employment_type']} | {doc.metadata['category']}\n{doc.page_content}\n")    