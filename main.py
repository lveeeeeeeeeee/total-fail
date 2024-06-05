import os
from dotenv import load_dotenv
from bot_util import get_mask
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

load_dotenv("vars.env")
TOKEN = os.environ["TOKEN"]

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command(commands=["start"]))
async def start(message: Message):
    await message.answer("Отправьте картинку шириной и высотой от 512 до 2048 и я покрашу на ней фон в зелёный (не покрашу тк я это очень плохо делаю)")


if __name__=="__main__":
    dp.run_polling(bot)