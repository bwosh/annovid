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

    def to_int(self):
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def clip(self, min_x:float=0.0, min_y:float=0.0, max_x:float=224.0, max_y:float=224.0):
        self.x1 = max(min(max_x, self.x1), min_x)
        self.y1 = max(min(max_y, self.y1), min_y)
        self.x2 = max(min(max_x, self.x2), min_x)
        self.y2 = max(min(max_y, self.y2), min_y)

    def iou(self, other_bbox:BBox):
        return 0.0 # TODO two boxes iou

    def __repr__(self):
        return f"BBox [({self.x1:0.2f},{self.y1:0.2f})-({self.x2:0.2f}),{self.y2:0.2f}) score={self.score:0.2f} class={self.class_id}/{self.class_name}] "