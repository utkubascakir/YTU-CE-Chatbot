import json
import pandas as pd
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from config.settings import *
from retrieval import YTUCEAssistant 


def run_evaluation(testset_path=TEST_QUESTIONS_PATH, output_path=RAGAS_RESULTS_PATH):
    print("[INFO] Loading test dataset...")
    with open(testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    eval_dataset = {
        "user_input": [],
        "reference": [],
        "response": [],
        "retrieved_contexts": []
    }

    # Set up the RAG bot
    bot = YTUCEAssistant()

    print("[INFO] Testing the bot with generated questions...")
    for i, item in enumerate(test_data):
        query = item["user_input"]
        reference = item["reference"]
        
        print(f"[{i+1}/{len(test_data)}] Asking bot: {query[:50]}...")
        bot_answer, bot_sources = bot.ask_bot(query)
        bot_contexts = [doc.page_content for doc in bot_sources]

        eval_dataset["user_input"].append(query)
        eval_dataset["reference"].append(reference)
        eval_dataset["response"].append(bot_answer)
        eval_dataset["retrieved_contexts"].append(bot_contexts)

    dataset = Dataset.from_dict(eval_dataset)

    print("[INFO] Setting up Ragas Evaluator LLM (Judges)...")
    eval_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0)
    eval_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)

    print("[INFO] Starting Evaluation. Scoring the bot...")
    safe_config = RunConfig(timeout=300, max_workers=1)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=safe_config
    )

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(result)
    
    df_result = result.to_pandas()
    df_result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[INFO] Detailed results saved to '{output_path}'.")


if __name__ == "__main__":
    run_evaluation()