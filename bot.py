import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import llama38b as llama38b  # tu módulo con generar_respuesta()


load_dotenv()  # carga las variables del .env
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Carpeta para guardar historiales de conversación
HISTORY_DIR = "./chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def cargar_historial(chat_id: int) -> str:
    path = os.path.join(HISTORY_DIR, f"{chat_id}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def guardar_historial(chat_id: int, user_text: str, bot_response: str) -> None:
    path = os.path.join(HISTORY_DIR, f"{chat_id}.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Usuario: {user_text}\nBot: {bot_response}\n")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hola! Soy NetAgent Bot. Envíame tus preguntas sobre Fortinet.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    user_text = update.message.text.strip()

    historial = cargar_historial(chat_id)

    # Limitar tamaño del historial para no saturar tokens
    max_historia_chars = 3000
    if len(historial) > max_historia_chars:
        historial = historial[-max_historia_chars:]

    # Construir prompt con contexto: incluye historial y el nuevo input
    prompt = f"{historial}Usuario: {user_text}\nBot:"

    # Generar respuesta desde el LLM (llama38b.py)
    respuesta = llama38b.generar_respuesta(prompt)

    # Guardar la interacción en el archivo de historial
    guardar_historial(chat_id, user_text, respuesta)

    # Responder al usuario
    await update.message.reply_text(respuesta)

def main():

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot iniciado...")
    app.run_polling()

if __name__ == "__main__":
    main()
