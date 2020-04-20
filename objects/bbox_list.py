import cv2
import numpy as np
import os

from objects.bbox import BBox
from objects.image import Image

class BBoxList:
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, bbox: BBox):
        self.data.append(bbox)

    def __getitem__(self, index):
        return self.data[index]

    def draw_on(self, img: Image, line_thickness:int=1, font_scale:int=0.3, line_height=10, pretty=True)->Image:
        img = img.to_rgb()

        line_type = cv2.LINE_AA if pretty else cv2.LINE_4

        for bbox in self.data:
            x1,y1,x2,y2 = bbox.to_int()

            cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,0), line_thickness, line_type)
            cv2.putText(img, f"{bbox.class_name}, {bbox.score:.2f}" , (x1+2,y1+line_height), cv2.FONT_HERSHEY_SIMPLEX , font_scale, (235,255,235), line_thickness, line_type)

        return Image.from_rgb_array(img)


    def __repr__(self):
        result = ""
        for i in self.data:
            result += str(i) + os.linesep
        return result


class BBoxListFrames:
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, bbox_list: BBoxList):
        self.data.append(bbox_list)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        counts = []
        for bbox_list in self.data:
            counts.append(len(bbox_list))
        result = f"BBoxListFrames with {len(counts)} frames with average of {np.mean(counts):0.2f} bbox(es) per frame."
        return result