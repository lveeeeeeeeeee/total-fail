import os
import io
from PIL import Image
from dotenv import load_dotenv
from bot_util import get_mask
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
import requests
import numpy as np
import cv2

load_dotenv("vars.env")
TOKEN = os.environ["TOKEN"]
GET_IMAGE = f"https://api.telegram.org/bot{TOKEN}/getFile?file_id="
GET_FILE = f"https://api.telegram.org/file/bot{TOKEN}"

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(F.photo)
async def photo_handler(message: Message):
    try:
        file_id = message.photo[-1].file_id
        response = requests.get(GET_IMAGE+file_id)
        img_path = response.json()['result']['file_path']
        img = requests.get(GET_FILE+f"/{img_path}").content
        save_name = f"everything/{message.chat.id}_{file_id}.png"
        with open(save_name, 'wb') as handler:
            handler.write(img)
        img = cv2.imread(save_name, cv2.IMREAD_COLOR)
        img = get_mask.predict_on_input(img)
        if len(img.shape) == 0:
            await message.answer("Размер изображения не подходит")
            return
        cv2.imwrite(save_name, img)
        await message.answer_photo(photo=FSInputFile(save_name))
        os.remove(save_name)
    except Exception as e:
        await message.answer("Произошла ошибка")
        print(e)
    finally:
        return

@dp.message(Command(commands=["start"]))
async def start(message: Message):
    await message.answer("Отправьте картинку шириной и высотой от 512 до 2048 и я покрашу на ней фон в зелёный (не покрашу тк я это очень плохо делаю)")

if __name__=="__main__":
    print("!!! START")
    dp.run_polling(bot)