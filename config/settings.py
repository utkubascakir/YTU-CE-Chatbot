from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
RERANK_MODEL = "rerank-multilingual-v3.0"

CHROMA_DB_PATH = str(BASE_DIR / "vector_db")
DATA_PATH = str(BASE_DIR / "data" / "docs")
TEST_QUESTIONS_PATH = str(BASE_DIR / "data" / "test_questions.json")
RAGAS_RESULTS_PATH = str(BASE_DIR / "evaluation_results.csv")