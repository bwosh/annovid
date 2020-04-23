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

    def reduce_to_classes(self, class_list):
        new_data = []
        for d in self.data:
            if d.class_name in class_list:
                new_data.append(d)
        self.data = new_data

    def append(self, bbox: BBox):
        self.data.append(bbox)

    def __getitem__(self, index):
        return self.data[index]

    def draw_on(self, img: Image, line_thickness:int=1, font_scale:int=0.3, line_height=10, pretty=True, additional_data=None)->Image:
        img = img.to_rgb()

        line_type = cv2.LINE_AA if pretty else cv2.LINE_4

        for bbox_idx, bbox in enumerate(self.data):
            x1,y1,x2,y2 = bbox.to_int()
            additional_text = ""

            if additional_data is not None:
                additional_text = ", ("+additional_data[bbox_idx]+")"

            cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,0), line_thickness, line_type)
            cv2.putText(img, f"{bbox.class_name}, {bbox.score:.2f}{additional_text}" , (x1+2,y1+line_height), cv2.FONT_HERSHEY_SIMPLEX , font_scale, (235,255,235), line_thickness, line_type)

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

    def reduce_to_classes(self, class_list):
        for d in self.data:
            d.reduce_to_classes(class_list)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        counts = []
        for bbox_list in self.data:
            counts.append(len(bbox_list))
        result = f"BBoxListFrames with {len(counts)} frames with average of {np.mean(counts):0.2f} bbox(es) per frame."
        return result