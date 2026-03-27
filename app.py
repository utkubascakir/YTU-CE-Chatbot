import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from src.retrieval.retrieval import YTUCEAssistant
from config.settings import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

bot = YTUCEAssistant()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = update.message.chat_id
    username = update.message.from_user.username or "Unknown"

    print(f"[INFO] Message from @{username} ({chat_id}): {user_message}")

    await update.message.chat.send_action("typing")

    answer, _ = bot.ask_bot(user_message)
    await update.message.reply_text(answer)


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
    print("[INFO] Starting Telegram bot...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    from telegram.ext import CommandHandler
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("[SUCCESS] Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()