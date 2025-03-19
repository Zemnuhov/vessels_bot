import datetime
import logging
import re
from pathlib import Path
import aiohttp
import cv2
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, PhotoSize, FSInputFile, InputMediaPhoto
from aiogram.filters import Command
from aiohttp import ClientError
from histoprocess._presentation.feature import Feature
from src.pydentic_models import PydenticPolygons
from src.stats_calculator import StatisticCalculator
from src.utils import draw_contours


class VesselsBot:
    def __init__(self, token: str, save_path: str, service_address: str):
        self.bot = Bot(token=token)
        self.sticker_id = (
            "CAACAgIAAxkBAAEM6RZnvZaVVW36nncAAZbsy7tSzbmyEG4AAiMAAygPahQnUSXnjCCkBjYE"
        )
        self.service_address = service_address
        self.dp = Dispatcher()
        self.router = Router()
        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir()
        self._setup_routes()
        logging.basicConfig(level=logging.INFO)

    def _setup_routes(self):
        self.router.message.register(self.cmd_start, Command("start"))
        self.router.message.register(self.handle_text, F.text)
        self.router.message.register(self.handle_photo, F.photo)
        self.dp.include_router(self.router)

    async def handle_text(self, message):
        nextcloud_pattern = (
            r"https?:\/\/(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:\/\S*)?"
        )
        if re.match(nextcloud_pattern, message.text):
            await self.handle_direct_link(message)
        else:
            await self.handle_nextcloud_file(message)

    async def cmd_start(self, message: Message):
        await message.answer(
            open("config/start_message.txt", "r").read(), parse_mode="MarkdownV2"
        )

    async def handle_nextcloud_file(self, message: Message):
        wsi_name = message.text.replace("/name ", "")
        await self.slide_response_pipline(
            message=message, wsi_name=wsi_name, link="/predict/nextcloud_file"
        )

    async def handle_direct_link(self, message: Message):
        wsi_name = Path(message.text).stem
        await self.slide_response_pipline(
            message=message, wsi_name=wsi_name, link="/predict/direct_link"
        )

    async def handle_photo(self, message: Message):
        sticker = await message.answer_sticker(self.sticker_id)
        photo_path = await self._save_photo(message=message)
        async with aiohttp.ClientSession() as session:
            with open(photo_path, "rb") as file:
                files = {"file": file}
                try:
                    async with session.post(
                        f"{self.service_address}/predict/image", data=files
                    ) as response:
                        if response.status == 200:
                            polygons = PydenticPolygons.model_validate(
                                await response.json()
                            ).as_polygons()
                            photo = cv2.cvtColor(
                                cv2.imread(str(photo_path)), cv2.COLOR_BGR2RGB
                            )
                            prediction_photo = draw_contours(polygons, photo)
                            prediction_photo_path = (
                                photo_path.parent / "prediction_image.png"
                            )
                            cv2.imwrite(f"{prediction_photo_path}", prediction_photo)
                            base_image = FSInputFile(str(photo_path))
                            pred_image = FSInputFile(str(prediction_photo_path))
                            media = [
                                InputMediaPhoto(
                                    media=base_image, caption="Original image"
                                ),
                                InputMediaPhoto(
                                    media=pred_image, caption="Prediction image"
                                ),
                            ]
                            await message.answer_media_group(media)
                            await message.answer(
                                StatisticCalculator()
                                .get_stats_from_polygons(polygons)
                                .to_string()
                            )
                            await message.delete()
                            await sticker.delete()
                        else:
                            await message.answer(
                                f"Произошла ошибка: {response.status}, {await response.text()}"
                            )
                            await message.delete()
                            await sticker.delete()
                except ClientError:
                    await sticker.delete()
                    await message.answer(f"Сервер недоступен")

    async def slide_response_pipline(self, message, wsi_name, link):
        sticker = await message.answer_sticker(self.sticker_id)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None)
        ) as session:
            try:
                async with session.post(
                    f"{self.service_address}{link}",
                    params={"file": str(message.text)},
                ) as response:
                    if response.status == 200:
                        resp = await response.json()
                        if "error" in dict(resp).keys():
                            await sticker.delete()
                            await message.answer(f"{dict(resp)["error"]}")
                            return None
                        polygons = PydenticPolygons.model_validate(
                            resp
                        ).as_polygons()
                        Feature.init().polygons_to_geojson(
                            polygons, str(self.save_path / "slides_result"), wsi_name
                        )
                        await message.answer_document(
                            FSInputFile(
                                str(
                                    self.save_path
                                    / "slides_result"
                                    / f"{wsi_name}.geojson"
                                )
                            )
                        )
                        await message.answer(
                            StatisticCalculator()
                            .get_stats_from_polygons(polygons)
                            .to_string()
                        )
                        await message.delete()
                        await sticker.delete()
                    else:
                        await message.answer(
                            f"Произошла ошибка: {response.status}, {await response.text()}"
                        )
                        await message.delete()
                        await sticker.delete()
            except ClientError:
                await sticker.delete()
                await message.answer(f"Сервер не доступен")

    async def _save_photo(self, message) -> Path:
        photo: PhotoSize = message.photo[-1]
        now = datetime.datetime.now()
        file_name = f'{message.chat.username}_{now.strftime("%Y_%m_%d_%H:%M:%S")}:{now.microsecond // 1000:03d}'
        file = await self.bot.get_file(photo.file_id)
        image_name = f"{file_name}.png"
        save_path = Path(self.save_path) / "images" / file_name / image_name
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        await self.bot.download(file, destination=(save_path))
        return save_path

    async def run(self):
        await self.dp.start_polling(self.bot)
