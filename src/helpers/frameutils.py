import timeit

import cv2
import numpy as np


class FramePreprocessor:
    """Crops and rescales an input image."""
    def __init__(self, scale: np.array, crop: np.array):
        top, right, bottom, left = crop
        scale_width, scale_height = scale

        self.process = lambda frame: cv2.resize(frame[top:-(bottom + 1), left:-(right + 1), :],
                                                None,
                                                fx=scale_width,
                                                fy=scale_height,
                                                interpolation=cv2.INTER_AREA)

    def __call__(self, frame: np.ndarray):
        return self.process(frame)


if __name__ == '__main__':
    preprocessor = FramePreprocessor(scale=[.5, .5], crop=[40, 3, 0, 3])
    print(preprocessor.process(np.zeros((240, 320, 3))).shape)
