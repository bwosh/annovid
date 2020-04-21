import matplotlib.pyplot as plt
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
        self.frame_preprocessing = None

    def set_frame_preprocessing(self, preprocessing):
        self.frame_preprocessing = preprocessing

    def annotate(self, path:str, cache_file_path:str=None, score_threshold=0.3, nms_threshold=0.5, maxframes=None)->BBoxListFrames:
        result = BBoxListFrames()

        if cache_file_path is not None:
            if os.path.isfile(cache_file_path):
                with open(cache_file_path,"rb") as file:
                    return pickle.load(file)

        reader = skvideo.io.FFmpegReader(path)
        (numframe, _, _, _) = reader.getShape()
        frame_idx=0
        for frame in tqdm(reader.nextFrame(), total=numframe):
            if maxframes is not None and frame_idx>=maxframes:
                break
            img = Image.from_rgb_array(frame)
            if self.frame_preprocessing is not None:
                img = self.frame_preprocessing(img)
            detections = self.annotator.get_bboxes(img, score_threshold=score_threshold, nms_threshold=nms_threshold)
            result.append(detections)

            if cache_file_path is not None:
                with open(cache_file_path,"wb") as file:
                    pickle.dump(result, file)

            frame_idx += 1

        return result

    def render(self, path:str, data:BBoxListFrames, output_path:str):
        reader = skvideo.io.FFmpegReader(path)
        writer = skvideo.io.FFmpegWriter(output_path)
        frame_idx = 0
        for frame in tqdm(reader.nextFrame(), total = len(data)):
            if frame_idx>=len(data):
                break
            detections = data[frame_idx]
            img = Image.from_rgb_array(frame)
            if self.frame_preprocessing is not None:
                img = self.frame_preprocessing(img)
            img = detections.draw_on(img).to_rgb()

            writer.writeFrame(img)
            frame_idx+=1
        writer.close()

    def get_bbox_heatmap(self, path:str, data:BBoxListFrames, use_sqrt = False):
        reader = skvideo.io.FFmpegReader(path)
        frame =  next(reader.nextFrame())
        if self.frame_preprocessing is not None:
            frame = self.frame_preprocessing(Image.from_rgb_array(frame)).to_rgb()
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

    def moving_average(self, data, length):
        cs = np.cumsum(data)
        mean100 = np.array((cs[length:]-cs[:-length])/length)
        trailing_zeros = np.repeat(np.NaN, length-1)
        return np.hstack([trailing_zeros, mean100])

    def get_count_plot(self, data:BBoxListFrames, moving_avg=2, title="", size=(15,8)):
        plt.figure(figsize=size)
        values = []
        for detections in data:
            values.append(len(detections))
        plt.plot(self.moving_average(values, moving_avg))
        plt.title(title)
        plt.savefig('plot_temp.png')
        return Image.open('plot_temp.png')
        
    # TODO groupping detections
    # TODO tracking lines
    # TODO applying insight video (plots on videos)