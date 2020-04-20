import numpy as np
import torch
import torchvision

from collections import defaultdict

from torchvision.models.detection import maskrcnn_resnet50_fpn

from base.annotators import BaseBBoxAnnotator
from objects.bbox import BBox
from objects.bbox_list import  BBoxList
from objects.image import Image

class MaskRCNNBBoxAnnotator(BaseBBoxAnnotator):
    def __init__(self):
        self.classes = {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motocycle",
            5: "airplane",
            6: "bus",
            7: "train",
            8: "truck"
        }

        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def to_tensor(self, img:Image)->torch.tensor:
        img = img.to_rgb()
        img = img.transpose(2,0,1)/255
        return torch.tensor(img,dtype=torch.float)

    def get_bboxes(self, image: Image, score_threshold:float=0.3, nms_threshold:float=0.5)->BBoxList:


        input = self.to_tensor(image).unsqueeze(0)
        outputs = self.model(input)
        output = outputs[0]

        boxes = output['boxes']
        labels = output['labels']
        scores = output['scores']

        # apply conditions
        detections = []
        for i in range(len(boxes)):
            label = int(labels[i])
            if label in self.classes.keys() and scores[i] >= score_threshold:
                detections.append( (boxes[i], scores[i], label) )
        del output
        del outputs

        # apply non-max supression
        boxes = torch.stack([c[0] for c in detections])
        scores = torch.stack([c[1] for c in detections])
        nms_result = torchvision.ops.nms(boxes, scores, nms_threshold)

        # gather final result
        result = BBoxList()
            
        for i in nms_result:
            x1,y1,x2,y2  = boxes[i].detach().cpu().numpy()
            label_id = detections[i][2]
            bbox = BBox(x1, y1, x2, y2, label_id,  self.classes[label_id], float(scores[i]))
            result.append(bbox)

        return result