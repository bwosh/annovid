import numpy as np
import skvideo.io
import os
import pickle

from tqdm import tqdm

from objects.bbox_list import BBoxListFrames
from objects.image import Image

class VideoAnnotator:
    def __init__(self, annotator):
        self.annotator = annotator

    def annotate(self, path:str, cache_file_path:str=None, score_threshold=0.3, nms_threshold=0.5)->BBoxListFrames:
        result = BBoxListFrames()

        if cache_file_path is not None:
            if os.path.isfile(cache_file_path):
                with open(cache_file_path,"rb") as file:
                    return pickle.load(file)

        reader = skvideo.io.FFmpegReader(path)
        for frame in tqdm(reader.nextFrame()):
            img = Image.from_rgb_array(frame)
            detections = self.annotator.get_bboxes(img, score_threshold=score_threshold, nms_threshold=nms_threshold)
            result.append(detections)

            if cache_file_path is not None:
                with open(cache_file_path,"wb") as file:
                    pickle.dump(result, file)

        return result

    def render(self, path:str, data:BBoxListFrames, output_path:str):
        reader = skvideo.io.FFmpegReader(path)
        writer = skvideo.io.FFmpegWriter(output_path)
        frame_idx = 0
        for frame in tqdm(reader.nextFrame()):
            if frame_idx>=len(data):
                break
            detections = data[frame_idx]
            img = Image.from_rgb_array(frame)
            img = detections.draw_on(img).to_rgb()

            writer.writeFrame(img)
            frame_idx+=1
        writer.close()

    def get_bbox_heatmap(self, path:str, data:BBoxListFrames, use_sqrt = False):
        reader = skvideo.io.FFmpegReader(path)
        frame =  next(reader.nextFrame())
        h,w,_ = frame.shape

        map = np.zeros((h,w,3), dtype=float)
        frame_idx = 0

        for frame_idx in range(len(data)):
            detections = data[frame_idx]
            for bbox in detections:
                x1,y1,x2,y2 = bbox.to_int()
                map[y1:y2,x1:x2,:] += 1.0

        map = map/np.max(map)
        if use_sqrt:
            map = np.sqrt(map)
            map = map/np.max(map)
        map *=255
        map = map.astype('uint8')
        return Image.from_rgb_array(map)
        

    # TODO bboxes persisiting on disk (caching)
    # TODO groupping detections -> utils?