import os, io
from dotenv import load_dotenv
from loguru import logger
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import whisper

load_dotenv()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)


async def audio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle received audio."""
    audio_file = update.message.audio or update.message.voice  # Get audio from the message

    if audio_file:
        file = await context.bot.get_file(audio_file.file_id)
        filepath = os.path.join("audios", f"{audio_file.file_id}.ogg")
        await file.download_to_drive(filepath)
        logger.debug(f"file {filepath} downloaded")

        model = whisper.load_model("base")

        audio = whisper.load_audio(filepath)
        audio = whisper.pad_or_trim(audio)
        logger.debug("Audio loaded")

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # decode the audio
        options = whisper.DecodingOptions(fp16=False, language="es")
        result = whisper.decode(model, mel, options)

        await update.message.reply_text(result.text)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.getenv("TOKEN")).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # handle audio
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, audio_handler))

    # on non command i.e message - echo the message on Telegram
    # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
