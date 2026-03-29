# YTU-CE Telegram Assistant

A Retrieval-Augmented Generation (RAG) based Telegram bot that answers questions about the Yıldız Technical University Computer Engineering department — including course information, university regulations, salary statistics, and student reviews.

---

## Features

- **Telegram Bot Interface** — Interact with the assistant directly via Telegram
- **Multi-source Ingestion** — Processes PDF regulations, Markdown files, and JSON-formatted student comments
- **Hybrid Search** — Combines ChromaDB vector search with BM25 keyword retrieval and multi-query expansion
- **Reranking** — Cohere reranker for improved result relevance
- **Conversation Memory** — Stores chat history in SQLite and rewrites follow-up questions for context-aware retrieval
- **Evaluation Pipeline** — RAGAS-based evaluation measuring context precision, recall, and faithfulness

---

## Project Structure

```text
chatbot/
├── data/                   # Raw documents (PDF, Markdown, JSON)
├── src/
│   ├── ingestion/          # Data processing and vector DB creation
│   ├── retrieval/          # Core RAG logic, query analysis, LLM interaction
│   └── evaluation/         # RAGAS evaluation scripts
├── config/                 # Configuration and settings
├── app.py                  # Telegram bot entrypoint
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Prerequisites

You will need API keys for the following services:

| Service | Purpose |
|---|---|
| [Google Gemini](https://aistudio.google.com/) | LLM and embeddings |
| [Cohere](https://cohere.com/) | Reranking |
| [Telegram BotFather](https://t.me/BotFather) | Bot token |

---

## Installation

### Option A: Docker (Recommended)

**Requirements:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# 1. Clone the repository
git clone https://github.com/utkubascakir/YTU-CE-Chatbot.git
cd YTU-CE-Chatbot

# 2. Set up environment variables
cp .env.example .env
# Fill in your API keys in .env

# 3. Build the Docker image
docker compose build

# 4. Create the vector database (first time only)
docker compose run --rm chatbot python -m src.ingestion.pipeline

# 5. Start the bot
docker compose up -d chatbot

# 6. View logs
docker compose logs -f chatbot
```

To stop the bot:
```bash
docker compose down
```

To restart without rebuilding (vector DB is preserved on your local disk):
```bash
docker compose up -d chatbot
```

---

### Option B: Local Setup

**Requirements:** Python 3.13+

```bash
# 1. Clone the repository
git clone https://github.com/utkubascakir/YTU-CE-Chatbot.git
cd YTU-CE-Chatbot

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Fill in your API keys in .env

# 5. Create the vector database (first time only)
python -m src.ingestion.pipeline

# 6. Start the bot
python app.py
```

---

## Environment Variables

```env
TELEGRAM_BOT_TOKEN=
GEMINI_API_KEY=
COHERE_API_KEY=
```

---

## Evaluation

To evaluate the RAG system's performance using the RAGAS framework:

```bash
# Local
python -m src.evaluation.evaluate

# Docker
docker compose run --rm chatbot python -m src.evaluation.evaluate
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Framework | LangChain |
| Vector Database | ChromaDB |
| LLM & Embeddings | Google Gemini |
| Reranking | Cohere |
| Evaluation | RAGAS |
| Bot Platform | python-telegram-bot |
| Conversation Storage | SQLite |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.