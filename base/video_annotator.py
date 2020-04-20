from objects.bbox import BBoxListFrames

class VideoAnnotator:
    def __init__(self, annotator):
        self.annotator = annotator

    def annotate(self, path:str)->BBoxListFrames:
        result = BBoxListFrames()
        # TODO process bboxes
        return result

    # TODO bboxes persisiting on disk (caching)

    # TODO groupping detections -> utils?