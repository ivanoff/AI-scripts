
"""
This script implements a Telegram bot that generates music samples based on text prompts using the Facebook MusicGen model.

Modules:
    os: Provides a way of using operating system dependent functionality.
    telebot: A Python library for the Telegram Bot API.
    scipy: A Python library used for scientific and technical computing.
    subprocess: Allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
    logging: Provides a way to configure logging for the script.
    datetime: Supplies classes for manipulating dates and times.
    transformers: A library for state-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.

Functions:
    send_welcome(message): Sends a welcome message when the /start command is received.
    send_file(message): Generates a music sample based on the received text prompt and sends the generated file back to the user.

Variables:
    synthesiser: A pipeline object for text-to-audio generation using the Facebook MusicGen model.
    BOT_TOKEN: The token for accessing the Telegram Bot API.
    FILES_DIRECTORY: The directory where generated files are stored.
    log_filename: The name of the log file.
    bot: The TeleBot object for interacting with the Telegram Bot API.
"""

import os
import telebot
import scipy
import subprocess
import os
import logging
from datetime import datetime
from transformers import pipeline

synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

BOT_TOKEN = '7.....2:A...........A'
FILES_DIRECTORY = './files'

log_filename = "server.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hi! Send me music prompt and I'll generate the sample")

@bot.message_handler(func=lambda message: True)
def send_file(message):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filenameWav = os.path.join(FILES_DIRECTORY, f"{timestamp}.wav")
    filename = os.path.join(FILES_DIRECTORY, f"{timestamp}.mp3")

    prompt = message.text.strip()

    filenamePrompt = os.path.join(FILES_DIRECTORY, os.path.normpath(prompt))
    if os.path.exists(filenamePrompt):
        with open(filenamePrompt, 'rb') as file:
            bot.send_document(message.chat.id, file)
    else:
        bot.reply_to(message, "Got it, starting to generate...")
        logging.info(f"{filename}: {prompt}")

        music = synthesiser(prompt, forward_params={"do_sample": True})
        scipy.io.wavfile.write(filenameWav, rate=music["sampling_rate"], data=music["audio"])
        subprocess.call(['ffmpeg', '-i', filenameWav, filename])
        os.remove(filenameWav)

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                bot.send_document(message.chat.id, file)
        else:
            bot.reply_to(message, f"File {filename} not found")

bot.polling(none_stop=True)
