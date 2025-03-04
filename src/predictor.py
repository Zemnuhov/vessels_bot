from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
from histomark_lib.torch_lib.models import Model
from histomark_lib.torch_lib.prediction import torch_predict
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from histoprocess import AnnotationType
from histoprocess._domain.model.polygon import Polygons
from histoprocess._presentation.feature import Feature
from histoprocess._presentation.image.image_grid_feature import ImageGridFeature
from histoprocess._presentation.image.image_inference_collection import (
    ImageInferenceCollection,
)
from histoprocess.transforms import PatchTransformer
from torch.utils.data import DataLoader

from src.data.dataloader_factory import DataBuilder, GridType
from src.data.image_dataset import ImageDataset
from src.postprocess_preds import (
    SegmentationPostprocess,
    InvasionSegmentationPostprocess,
)


class ImagePredictor:

    def __init__(
        self, segmentation_model: Model, invasion_model: Model, device: str = "cpu"
    ):
        self.segmentation_model = segmentation_model
        self.invasion_model = invasion_model
        self.device = device
        self.segmentation_transforms = [
            albumentations.Resize(width=512, height=512),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(transpose_mask=True),
        ]
        self.invasion_transforms = [
            albumentations.Resize(width=256, height=256),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(transpose_mask=True),
        ]

    def predict_image(self, image: Union[Path, List[Path]]):
        image_paths = [image] if isinstance(image, Path) else image
        for img_path in image_paths:
            segmentation_data_builder = DataBuilder(
                img_path,
                grid_type=GridType.FULL,
                transforms=self.segmentation_transforms,
            )
            segmentation_predict = torch_predict(
                model=self.segmentation_model,
                data=segmentation_data_builder.build(),
                accelerator=self.device,
                binary_threshold=0.5,
            )
            segmentation_predict = SegmentationPostprocess().execute(
                segmentation_predict, segmentation_data_builder.grid
            )
            Feature.init().patches_to_geojson(
                segmentation_predict, segmentation_data_builder.grid, str("images/")
            )

            invasion_data_builder = DataBuilder(
                img_path,
                grid_type=GridType.ANNOTATION,
                transforms=self.invasion_transforms,
                wsa_path="images/annotation.geojson",
                is_fill_without_mask=True,
            )

            invasion_predict = torch_predict(
                model=self.invasion_model,
                data=invasion_data_builder.build(),
                accelerator=self.device,
                binary_threshold=0.995,
            )
            invasion_predict = InvasionSegmentationPostprocess().execute(
                invasion_predict, invasion_data_builder.grid
            )
            Feature.init().patches_to_geojson(
                invasion_predict,
                invasion_data_builder.grid,
                str("images/"),
                file_name="invasion",
            )
            Feature.init().combine_prediction_results(
                segmentation_geojson_path=str("images/annotation.geojson"),
                classification_geojson_path=str("images/invasion.geojson"),
                classification_annotation_type=["Invasion"],
                save_path="images/",
            )
            poly = Feature.init().get_polygons_from_wsa(
                "images/combine_prediction.geojson"
            )
            res = self.draw_contours(
                poly, cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            )
            return res

    def draw_contours(self, annotations: Polygons, image: np.ndarray):
        image_copy = image.copy()
        image_copy2 = image.copy()
        for polygon in annotations.value:
            coord = tuple(np.array([polygon.coordinates]).astype(np.int32))
            cv2.drawContours(
                image=image_copy,
                contours=coord,
                contourIdx=-1,
                color=(
                    (0, 255, 0)
                    if polygon.annotation_type == AnnotationType.CLEAN_VESSEL
                    else (255, 0, 0)
                ),
                thickness=cv2.FILLED,
                lineType=cv2.LINE_4,
            )
            cv2.drawContours(
                image=image_copy2,
                contours=coord,
                contourIdx=-1,
                color=(
                    (0, 255, 0)
                    if polygon.annotation_type == AnnotationType.CLEAN_VESSEL
                    else (255, 0, 0)
                ),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        image_copy = cv2.addWeighted(image_copy2, 0.9, image_copy, 0.2, 0.0)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        return image_copy
