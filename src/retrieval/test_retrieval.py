from src.retrieval.retrieval import YTUCEAssistant
from config.settings import GEMINI_API_KEY


def main():
    bot = YTUCEAssistant()

    while True:
        query = input("\nEnter your question (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
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
            source_type = doc.metadata.get("source_type", "Unknown")
            doc_type = doc.metadata.get("doc_type", "Unknown")
            print(f"{idx+1}. {source_type}/{doc_type} | {doc.page_content[:150]}...")


if __name__ == "__main__":
    main()