import os

class BBox:
    def __init__(self, x1:float, y1:float, x2:float, y2:float, class_id:int, class_name:str, score:float):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.class_id = class_id 
        self.class_name = class_name 
        self.score = score

    def equals(self, other_bbox):
        return self.x1 == other_bbox.x1 and \
            self.x2 == other_bbox.x2 and \
            self.y1 == other_bbox.y1 and \
            self.y2 == other_bbox.y2 and \
            self.class_id == other_bbox.class_id and \
            self.class_name == other_bbox.class_name and \
            self.score == other_bbox.score


    def to_int(self):
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def clip(self, min_x:float=0.0, min_y:float=0.0, max_x:float=224.0, max_y:float=224.0):
        self.x1 = max(min(max_x, self.x1), min_x)
        self.y1 = max(min(max_y, self.y1), min_y)
        self.x2 = max(min(max_x, self.x2), min_x)
        self.y2 = max(min(max_y, self.y2), min_y)

    @property
    def center(self):
        return (self.x1+(self.x2-self.x1)/2,self.y1+(self.y2-self.y1)/2)

    def iou(self, other_bbox):
        # calculate intersection
        xA = max(self.x1, other_bbox.x1)
        yA = max(self.y1, other_bbox.y1)
        xB = min(self.x2, other_bbox.x2)
        yB = min(self.y2, other_bbox.y2)

        intersection_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if intersection_area == 0:
            return 0

        # calculate boxes areas
        area1 = abs((self.x2 - self.x1) * (self.y1 - self.y1))
        area2 = abs((other_bbox.x2 - other_bbox.x1) * (other_bbox.y2 - other_bbox.y1))

        iou = intersection_area / float(area1 + area2 - intersection_area)
        return iou

    def __repr__(self):
        return f"BBox [({self.x1:0.2f},{self.y1:0.2f})-({self.x2:0.2f},{self.y2:0.2f}) score={self.score:0.2f} class={self.class_id}/{self.class_name}] "