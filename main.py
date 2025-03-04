import asyncio
import logging
from pathlib import Path

import cv2
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, PhotoSize, FSInputFile, InputMediaPhoto
from aiogram.filters import Command
from histomark_lib.torch_lib import torch_predict
from histomark_lib.torch_lib.models import SMPModel

from src.predictor import ImagePredictor

TOKEN = "7988426542:AAEoxlI3HtI_hg7b4KzuZI1L47DgHmH1XNI"

bot = Bot(token=TOKEN)
dp = Dispatcher()
router = Router()
logging.basicConfig(level=logging.INFO)

image_predictor = ImagePredictor(
    segmentation_model=SMPModel(
        encoder_weights="imagenet",
        in_channels=3,
        model_name="segformer",
        encoder_name="mit_b3",
        classes=2,
        checkpoint_path="/home/egor/programm/histomark_new_template/histomark/logs/train/runs/2025-02-27_16-27-50/checkpoints/clear_model_epoch:029-val_loss:0.0160-val_fbeta:0.9861-val_IoU:0.9688-val_AucROC:0.9893.ckpt",
    ),
    invasion_model=SMPModel(
        encoder_weights="imagenet",
        in_channels=3,
        model_name="segformer",
        encoder_name="timm-efficientnet-b3",
        classes=1,
        checkpoint_path="/home/egor/programm/histomark_new_template/histomark/logs/train/runs/2025-02-10_17-31-05/checkpoints/clear_model_epoch:021-val_loss:0.1564-val_fbeta:0.9112-val_IoU:0.8368-val_AucROC:0.9844.ckpt",
    ),
    device="cuda:2"
)

save_path = "images/"


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Привет! Я ваш Telegram-бот.")


@router.message(F.text)
async def echo(message: Message):
    await message.answer(f"Вы сказали: {message.text}")


@router.message(F.photo)
async def handle_photo(message: Message):
    photo: PhotoSize = message.photo[-1]
    file_id = photo.file_id
    file = await bot.get_file(file_id)
    image_name = f"{file_id}.png"
    await bot.download(file, destination=str(Path(save_path) / image_name))
    try:
        image = image_predictor.predict_image(Path(save_path) / image_name)
        cv2.imwrite("images/result.png", image)

        base_image = FSInputFile(str(Path(save_path) / image_name))
        pred_image = FSInputFile(f"images/result.png")
        media = [
            InputMediaPhoto(media=base_image, caption="Base_image"),
            InputMediaPhoto(media=pred_image, caption="Pred_image")
        ]
        await message.answer_media_group(media)

    except Exception():
        await message.answer("Что то пошло не так")
    await message.delete()


async def main():
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
