import numpy as np

from objects.bbox import BBox
from objects.bbox_list import BBoxList
from objects.image import Image

class BaseBBoxAnnotator:
    def __init__(self):
        pass

    def get_bboxes(self, image: Image)->BBoxList:
        raise Exception("Not implemented")
