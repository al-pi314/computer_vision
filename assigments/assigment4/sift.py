import cv2
import numpy as np


class Sift:
    def __init__(self, octaves=4, levels=3, sigma0=2, sigmaS=1.6):
        self.octaves = octaves
        self.levels = levels
        self.sigma0 = sigma0
        self.sigmaS = sigmaS
    
    def _gaussian_kernel(self, sigma):
        size = int(6 * sigma)
        if size % 2 == 0:
            size += 1
        return cv2.getGaussianKernel(size, sigma)

    def _gaussian_filter(self, image, sigma):
        kernel =self._gaussian_kernel(sigma)
        return cv2.sepFilter2D(image, ddepth=-1, kernelX=kernel, kernelY=kernel)

    def _gaussian_pyramid(self, image):
        pyramid = []
        lvl_sigma = self.sigma0
        for _ in range(self.levels +1):
            pyramid.append(self._gaussian_filter(image, lvl_sigma))
            lvl_sigma *= self.sigmaS
        return pyramid

    def _dog_pyramid(self, image):
        gaussian_pyramid = self._gaussian_pyramid(image)

        pyramid = np.empty((self.levels, *image.shape))
        for i in range(self.levels):
            pyramid[i] = (gaussian_pyramid[i + 1] - gaussian_pyramid[i])
        return pyramid

    def _maxima_in_scale_space(self, scale_space):
        maxima = np.zeros_like(scale_space)
        for i in range(1, len(scale_space) -1):
            for y in range(1, len(scale_space[i]) - 1):
                for x in range(1, len(scale_space[i][y]) - 1):
                    if self._is_unique_maximum(maxima, scale_space, i, x, y):
                        maxima[i, y, x] = scale_space[i, y, x]
        return maxima

    def _is_unique_maximum(self, maxima, scale_space, i, x, y):
        local_cube = scale_space[i - 1:i + 2, y - 1:y + 2, x - 1:x + 2]
        other_maxima = maxima[i - 1:i + 2, y - 1:y + 2, x - 1:x + 2]
        return scale_space[i, y, x] == local_cube.max() and other_maxima.max() == 0

    def _threshold(self, values, threshold):
        _v = values.copy()
        _v[values < threshold] = 0
        return _v

    def detect_keypoints(self, I, threshold):
        scale_space = self._dog_pyramid(I)
        maxima = self._maxima_in_scale_space(scale_space)
        maxima_th = self._threshold(maxima, threshold)
        return np.argwhere(maxima_th > 0)
        
    
