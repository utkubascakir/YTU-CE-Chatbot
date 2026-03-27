# YTU-CE Telegram Assistant (YTUCEAssistant)

This project is a Retrieval-Augmented Generation (RAG) system deployed as a Telegram bot. It provides accurate answers to questions about the Yıldız Technical University Computer Engineering (YTUCE) department, student comments, and university regulations.

## 🚀 Features

- **Telegram Bot Interface:** Interact with the assistant directly via Telegram.
- **Multi-source Data Ingestion:** Processes PDF regulations, Markdown files (salary statistics), and student comments in JSON format.
- **Advanced Retrieval:** Uses ChromaDB as a vector database for efficient similarity search.
- **LLM Integration:** Powered by Google Gemini models for natural language understanding and generation.
- **Reranking:** Integrated with Cohere for better search result relevance (optional/configurable).
- **Evaluation:** Includes an evaluation pipeline using the `Ragas` framework to measure context precision, recall, and faithfulness.

## 📂 Project Structure

```text
chatbot/
├── data/               # Raw documents (PDF, MD, JSON)
├── src/
│   ├── ingestion/      # Scripts for processing and indexing data
│   ├── retrieval/      # Core RAG logic and LLM interaction
│   └── evaluation/     # RAGAS evaluation scripts
├── config/             # Configuration files
├── app.py              # Telegram Bot application
├── requirements.txt    # Project dependencies
└── .gitignore          # Git ignore rules
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/utkubascakir/YTU-CE-Chatbot.git
   cd YTU-CE-Chatbot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   COHERE_API_KEY=your_cohere_api_key
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   ```

## 📖 Usage

### 1. Data Ingestion
Before running the assistant, you need to process the documents and create the vector database:
```bash
python -m src.ingestion.pipeline
```

### 2. Run the Telegram Bot
Start the bot application:
```bash
python app.py
```

### 3. Evaluation
To evaluate the RAG system's performance:
```bash
python -m src.evaluation.evaluate
```

## 🧪 Technologies Used

- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **LLM:** Google Gemini
- **Reranking:** Cohere
- **Evaluation:** Ragas
- **Bot Platform:** python-telegram-bot
- **Data Handling:** Pandas, Pydantic

## 📝 License
This project is for educational purposes.
