import cv2
import numpy as np

from PIL import Image as PilImage

from objects.bbox import BBox 

class Image:
    def __init__(self):
        self._initialized = False
        self._pil_image = None
        self._rgb_image = None
        self._bgr_image = None
        self._mode = None

    @staticmethod
    def open(path):
        img = Image()
        img._initialized = True
        img._mode = "BGR" 
        img._bgr_image = cv2.imread(path)       
        return img

    @staticmethod
    def from_rgb_array(rgb_array: np.ndarray, copy:bool=False):
        img = Image()
        img._initialized = True
        img._mode = "RGB"

        if copy:
            img._rgb_image = rgb_array.copy()
        else:
            img._rgb_image = rgb_array

        return img

    @staticmethod
    def from_bgr_cv2_image(bgr_image: np.ndarray, copy:bool=False):
        img = Image()
        img._initialized = True
        img._mode = "BGR"

        if copy:
            img._bgr_image = bgr_image.copy()
        else:
            img._bgr_image = bgr_image

        return img
    
    @staticmethod
    def from_pil(pil_image: PilImage, copy:bool=False):
        img = Image()
        img._initialized = True
        img._mode = "PIL"

        if copy:
            img._pil_image = pil_image.copy()
        else:
            img._pil_image = pil_image

        return img

    def preview(self, thumbnail_size=None):
        if not self._initialized:
            raise Exception("Not initialized")

        if self._mode == "BGR":
            result = PilImage.fromarray(self._bgr_image[:,:,::-1])
        elif self._mode == "RGB":
            result = PilImage.fromarray(self._rgb_image)
        else:
            result = self._pil_image.copy()

        if not thumbnail_size is None:
            result.thumbnail(thumbnail_size)

        return result

    def to_rgb(self):
        if self._mode == "PIL":
            return self._pil_image.toarray()
        elif self._mode == "BGR":
            return self._bgr_image[:,:,::-1].copy()
        
        return self._rgb_image.copy()

    @property
    def pp(self):
        return self.preview(thumbnail_size=(200,200))

    def get_crop(self, offset=(0,0), source_size=(224,224), target_size=None):
        ox, oy = offset
        w, h = source_size
        if target_size is None:
            target_size = source_size
        target_w, target_h = target_size

        img = self.to_rgb()
        result = img[oy:oy+h, ox:ox+w,:].copy()
        result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
        return Image.from_rgb_array(result), BBox(ox, oy, ox+w, oy+h, -1, "CROP", 1.0)

    # TODO merge images (one big image with overview)