from tqdm import tqdm

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
        self.bbox_history = self.bbox_history[-self.history_length:]
        # TODO add interpolation for missing frames ?

    def get_total_occurences(self):
        return len(self.full_bbox_history)

    def contains(self, frame_index:int, bbox:BBox):
        for bb, idx in self.full_bbox_history:
            if idx==frame_index and bb.equals(bbox):  
                return True
            if idx > frame_index:
                return False
        return False

    def get_history(self, frame_index:int, bbox:BBox, history_length:int):
        result = []
        for hist_index, (bb, idx) in enumerate(self.full_bbox_history):
            if idx==frame_index and bb.equals(bbox): 
                start = max(0, hist_index-history_length) 
                return self.full_bbox_history[start:hist_index+1]
        return result

    def get_match_score(self, frame_index:int, other_bbox:BBox, allowed_history_length=1):
        loops = min(allowed_history_length, len(self.bbox_history))

        for i in range(loops):
            historical_bbox, historical_frame  = self.bbox_history[-i]

            if historical_frame<frame_index-allowed_history_length:
                iou = historical_bbox.iou(other_bbox)
                if iou>0.0:
                    return iou
            else:
                return 0.0
        return 0.0

class KnownBBoxList:
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, bbox_list: KnownBBox):
        self.data.append(bbox_list)

    def __getitem__(self, index):
        return self.data[index]

    def get_group_id(self, frame_index:int, bbox:BBox, history_length = None):
        for g_idx, group in enumerate(self.data):
            if group.contains(frame_index, bbox):
                if history_length is None:
                    return g_idx
                else:
                    return g_idx, group.get_history(frame_index, bbox, history_length)
        return None

    def __repr__(self):
        result = f"KnownBBoxList with {len(self.data)} bboxes"
        return result

class BBoxGroupping:
    def __init__(self, frames_with_detections:BBoxListFrames):
        self.data = frames_with_detections

    def group(self, iou_threshold:float, fill_missing_frames:int=0)->BBoxListFrames:
        history_length = fill_missing_frames+1

        # Gether boxes
        known_boxes = KnownBBoxList()
        for frame_id, frame_with_detections in enumerate(tqdm(self.data, desc="Groupping")):
            for bbox_idx, bbox in enumerate(frame_with_detections):
                # find matching known box
                matching_known_box_index = None

                match_score = 0.0
                best_score = 0.0
                for kb_index, kb in enumerate(known_boxes):
                    match_score = kb.get_match_score(frame_id, bbox, history_length)
                    if match_score > iou_threshold and match_score>best_score:
                        matching_known_box_index = kb_index
                        best_score = match_score

                if matching_known_box_index is not None:
                    known_boxes[matching_known_box_index].add_new_apperance(frame_id, bbox)
                else:
                    new_known_bbox = KnownBBox(frame_id, bbox, history_length)
                    known_boxes.append(new_known_bbox)

        return known_boxes