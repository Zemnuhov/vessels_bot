from typing import List, Union, Optional

from histoprocess import AnnotationType
from histoprocess._domain.model.polygon import GeometryType, Polygon, Polygons
from pydantic import BaseModel


class PydenticPolygon(BaseModel):
    geometry_type: GeometryType
    annotation_type: AnnotationType
    color: Union[List, int]
    coordinates: List[List]
    name: Optional[str]

    def as_polygon(self) -> Polygon:
        return Polygon(
            geometry_type=self.geometry_type,
            annotation_type=self.annotation_type,
            color=self.color,
            coordinates=self.coordinates,
            name=self.name,
        )


class PydenticPolygons(BaseModel):
    value: List[PydenticPolygon]

    def as_polygons(self) -> Polygons:
        return Polygons(value=[PydenticPolygon.as_polygon(poly) for poly in self.value])
