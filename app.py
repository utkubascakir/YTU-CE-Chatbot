import sqlite3
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler
from src.retrieval.retrieval import YTUCEAssistant
from config.settings import TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, GEMINI_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

bot = YTUCEAssistant()
rewrite_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# SQLite setup
def init_db():
    conn = sqlite3.connect("conversations.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_message(chat_id: int, role: str, content: str):
    conn = sqlite3.connect("conversations.db")
    conn.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content)
    )
    conn.commit()
    conn.close()


def get_history(chat_id: int, limit: int = 4) -> list:
    conn = sqlite3.connect("conversations.db")
    cursor = conn.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
        (chat_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    # Reverse to get chronological order
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


def rewrite_query(history: list, user_query: str) -> str:
    if not history:
        return user_query

    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" for msg in history
    ])

    prompt = (
        f"Aşağıdaki konuşma geçmişine ve kullanıcının son sorusuna bak.\n\n"
        f"Geçmiş:\n{history_text}\n\n"
        f"Son soru: {user_query}\n\n"
        f"Eğer son soru geçmişe atıfta bulunuyorsa (örn: 'peki ya', 'o hocayı', 'onun maaşı' gibi), "
        f"geçmişten bağımsız, tek başına anlamlı bir soruya dönüştür. "
        f"Eğer soru zaten bağımsızsa olduğu gibi döndür. "
        f"Sadece soruyu yaz, başka hiçbir şey yazma."
    )

    response = rewrite_llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip()
    print(f"[INFO] Query rewritten: '{user_query}' -> '{rewritten}'")
    return rewritten


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = update.message.chat_id
    username = update.message.from_user.username or "Unknown"

    print(f"[INFO] Message from @{username} ({chat_id}): {user_message}")
    await update.message.chat.send_action("typing")

    # Get history and rewrite query if needed
    history = get_history(chat_id)
    final_query = rewrite_query(history, user_message)

    answer, _ = bot.ask_bot(final_query)

    # Save to SQLite
    save_message(chat_id, "user", user_message)
    save_message(chat_id, "assistant", answer)

    await update.message.reply_text(answer, parse_mode="HTML")


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "Merhaba! Ben YTÜ Bilgisayar Mühendisliği Asistanıyım 🎓\n\n"
        "Sana şu konularda yardımcı olabilirim:\n"
        "• Ders ve müfredat bilgileri\n"
        "• Yönetmelik ve yönergeler\n"
        "• Mezun maaş istatistikleri\n"
        "• Hoca yorumları\n\n"
        "Sorunuzu yazabilirsiniz!"
    )
    await update.message.reply_text(welcome)


def main():
    init_db()
    print("[INFO] Starting Telegram bot...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("[SUCCESS] Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()