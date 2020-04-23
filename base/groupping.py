from objects.bbox_list import BBoxListFrames
from objects.bbox import BBox

class KnownBBox:
    def __init__(self, frame_index:int, bbox:BBox, history_length:int):
        self.bbox_history = [(bbox, frame_index)]
        self.full_bbox_history = [(bbox, frame_index)]
        self.history_length = history_length

    def add_new_apperance(self, frame_index:int, bbox:BBox):
        new_entry = (bbox, frame_index)
        self.bbox_history.append(new_entry)
        self.full_bbox_history.append(new_entry)
        self.bbox_history = self.bbox_history[-self.history_length,:]

    def get_total_occurences(self):
        return len(self.full_bbox_history)

    def get_match_score(self, frame_index:int, other_bbox:BBox, allowed_history_length=1):
        loops = min(allowed_history_length, self.history_length)

        for i in range(loops):
            historical_bbox, historical_frame  = self.bbox_history[-i]

            if historical_frame<frame_index-allowed_history_length:
                iou = historical_bbox.iou(other_bbox)
                if iou>0.0:
                    return iou
            else:
                return 0.0
    return 0.0

class BBoxGroupping:
    def __init__(self, frames_with_detections:BBoxListFrames):
        self.data = frames_with_detections
        self.known_boxes = {}

    def group(self, iou_threshold:float, fill_missing_frames:int=0)->BBoxListFrames:
        result = BBoxListFrames()

        return result