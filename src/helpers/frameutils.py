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
    preprocessor = FramePreprocessor(scale=[1.0, 1.0], crop=[2, 2, 2, 2])
    print(timeit.timeit(lambda: preprocessor.process(np.zeros((10, 10, 3))), number=100000))
