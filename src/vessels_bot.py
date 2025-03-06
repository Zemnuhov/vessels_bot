import logging
from pathlib import Path
import cv2
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, PhotoSize, FSInputFile, InputMediaPhoto
from aiogram.filters import Command
from src.bot_event_handler import BotEventHandler
from src.predictors.image_predictor import ImagePredictor
from histomark_lib.torch_lib.models import Model
from src.stats_calculator import StatisticCalculator



class VesselsBot:
    def __init__(
        self,
        token: str,
        vessels_model: Model,
        invasion_model: Model,
        save_path: str = "images/",
        prediction_device: str = "cpu",
    ):
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.router = Router()
        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir()
        self.image_predictor = ImagePredictor(
            segmentation_model=vessels_model,
            invasion_model=invasion_model,
            device=prediction_device,
        )
        self._setup_routes()
        logging.basicConfig(level=logging.INFO)

    def _setup_routes(self):
        nextcloud_pattern = r"https:\/\/nextcloud\.novisp\.ru\/s\/[a-zA-Z0-9]+\/download\/[a-zA-Z0-9_]+\.(tif|tiff|svs)"
        self.router.message.register(self.cmd_start, Command("start"))
        self.router.message.register(self.echo, F.text)
        self.router.message.register(self.handle_nextcloud, F.text.regexp(nextcloud_pattern))
        self.router.message.register(self.handle_photo, F.photo)
        self.dp.include_router(self.router)

    async def cmd_start(self, message: Message):
        await message.answer("Привет! Я ваш Telegram-бот.")

    async def echo(self, message: Message):
        await message.answer(f"Вы сказали: {message.text}")

    async def handle_nextcloud(self, message: Message):
        await message.answer(f"Вы сказали: {message.text}")

    async def handle_photo(self, message: Message):
        sticker = await message.answer_sticker("CAACAgIAAxkBAAEM6RZnvZaVVW36nncAAZbsy7tSzbmyEG4AAiMAAygPahQnUSXnjCCkBjYE")
        #try:
        event_handler = BotEventHandler(self.bot, self.save_path, self.image_predictor)
        response = await event_handler.handle_photo_message(message=message)
        await message.answer_media_group(response.media)
        await message.answer(
            StatisticCalculator().get_stats_from_polygons(response.polygons).to_string()
        )
        #except Exception:
            #await message.answer("Что-то пошло не так")
        await message.delete()
        await sticker.delete()

    async def run(self):
        await self.dp.start_polling(self.bot)