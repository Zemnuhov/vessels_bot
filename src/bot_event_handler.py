from histoprocess._domain.model.polygon import Polygons
from typing import Any
from aiogram.types import Message, PhotoSize, FSInputFile, InputMediaPhoto
from dataclasses import dataclass
from pathlib import Path
from aiogram import Bot
import cv2
from histoprocess.feature import Feature
from histoprocess.filters import PolygonAreaFilter
from src.utils import draw_contours
from src.image_predictor import ImagePredictor


@dataclass
class HandlerResponse:
    media: Any
    polygons: Polygons


class BotEventHandler:

    def __init__(self, bot: Bot, data_path: Path, image_predictor: ImagePredictor):
        self.data_path = data_path
        self.bot = bot
        self.image_predictor = image_predictor

    async def _save_photo(self, message) -> Path:
        photo: PhotoSize = message.photo[-1]
        file_id = photo.file_id
        file = await self.bot.get_file(file_id)
        image_name = f"original_image.png"
        save_path = Path(self.data_path) / file_id / image_name
        if not save_path.parent.exists():
            save_path.parent.mkdir()
        await self.bot.download(file, destination=(save_path))
        return save_path

    async def handle_photo_message(self, message) -> HandlerResponse:
        photo_path = await self._save_photo(message=message)
        photo = cv2.cvtColor(cv2.imread(str(photo_path)), cv2.COLOR_BGR2RGB)
        polygons = self.image_predictor.predict(photo_path)
        polygons = Feature.init().filter_polygons_by_area(
            polygons=polygons,
            area_filter=PolygonAreaFilter(
                area_min=(photo.shape[0] * photo.shape[1]) * 0.001
            ),
        )
        prediction_photo = draw_contours(polygons, photo)
        prediction_photo_path = photo_path.parent / "prediction_image.png"
        cv2.imwrite(f"{prediction_photo_path}", prediction_photo)
        base_image = FSInputFile(str(photo_path))
        pred_image = FSInputFile(str(prediction_photo_path))
        response = HandlerResponse(
            media=[
                InputMediaPhoto(media=base_image, caption="Original image"),
                InputMediaPhoto(media=pred_image, caption="Prediction image"),
            ],
            polygons=polygons,
        )
        return response
