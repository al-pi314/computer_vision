import cv2
import numpy as np


class Sift:
    def __init__(self, octaves=4, levels=3, sigma0=2, sigmaS=1.6):
        self.octaves = octaves
        self.levels = levels
        self.sigma0 = sigma0
        self.sigmaS = sigmaS

    def load_image(self, image):
        self._img = image

        self._imgDx = cv2.Sobel(self._img, ddepth=-1, dx=1, dy=0)
        self._imgDy = cv2.Sobel(self._img, ddepth=-1, dx=0, dy=1)

        self._imgMag = np.sqrt(self._imgDx**2 + self._imgDy**2)
        self._imgAng = np.arctan2(self._imgDy, self._imgDx)
    
    def _gaussian_kernel(self, sigma):
        size = int(6 * sigma)
        if size % 2 == 0:
            size += 1
        return cv2.getGaussianKernel(size, sigma)

    def _gaussian_filter(self, sigma):
        kernel =self._gaussian_kernel(sigma)
        return cv2.sepFilter2D(self._img, ddepth=-1, kernelX=kernel, kernelY=kernel)

    def _gaussian_pyramid(self):
        pyramid = []
        lvl_sigma = self.sigma0
        for _ in range(self.levels +1):
            pyramid.append(self._gaussian_filter(lvl_sigma))
            lvl_sigma *= self.sigmaS
        return pyramid

    def _dog_pyramid(self):
        gaussian_pyramid = self._gaussian_pyramid()

        pyramid = np.empty((self.levels, *self._img.shape))
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

    def _reject_weak_points(self, keypoints, threshold):
        remaining = []
        for i, y, x in keypoints:
            if self._imgMag[y, x] > threshold:
                remaining.append(np.array([i, y, x]))
        return np.array(remaining)

    def _harris_response(self, sigma=2, alpha=0.06):
        C11 = cv2.GaussianBlur(self._imgDx**2, ksize=(0, 0), sigmaX=sigma)
        C12 = cv2.GaussianBlur(self._imgDx * self._imgDy, ksize=(0, 0), sigmaX=sigma)
        C22 = cv2.GaussianBlur(self._imgDy**2, ksize=(0, 0), sigmaX=sigma)
        
        detC = C11*C22 - C12**2
        traceC = C11 + C22

        return detC - alpha * (traceC**2)

    def _reject_edge_points(self, keypoints, threshold):
        h_response = self._harris_response()

        remaining = []
        for i, y, x in keypoints:
            if h_response[y, x] < threshold:
                remaining.append(np.array([i, y, x]))
        return np.array(remaining)

    def _get_orientation_for_keypoint(self, i, y, x, bins=36):
        if y < 8 or y > self._img.shape[0] - 8 or x < 8 or x > self._img.shape[1] - 8:
            return []

        window_mag = self._imgMag[y - 8:y + 8, x - 8:x + 8]
        window_ang = self._imgAng[y - 8:y + 8, x - 8:x + 8]

        histogram = np.zeros(bins)
        angle_step = 2 * np.pi / bins
        for wy in range(len(window_mag)):
            for wx in range(len(window_mag[wy])):
                shifted_ang = window_ang[wy, wx] + np.pi
                bin_idx = int(shifted_ang / angle_step)
                # angle of 2pi is equal to angle 0
                if bin_idx == bins:
                    bin_idx = 0
                histogram[bin_idx] += window_mag[wy, wx]
        histogram /= histogram.sum()

        strong_bins = np.argwhere(histogram > 0.8 * histogram.max()).flatten()

        angle = lambda idx: idx * angle_step
        strength = lambda idx: histogram[idx]
        keypoints = [[i, y, x, angle(idx), strength(idx)] for idx in strong_bins]
        return np.array(keypoints)

    def _get_orientations(self, keypoints):
        keypoints_with_orientation = []
        for i, y, x in keypoints:
            keypoint_orientations = self._get_orientation_for_keypoint(i, y, x)
            keypoints_with_orientation.extend(keypoint_orientations)
        return np.array(keypoints_with_orientation)

    def detect_keypoints(self, w_th, e_th):
        scale_space = self._dog_pyramid()

        maxima = self._maxima_in_scale_space(scale_space)
        keypoints = np.argwhere(maxima > 0)

        robust_kp = self._reject_weak_points(keypoints, w_th)
        robust_kp = self._reject_edge_points(robust_kp, e_th)

        return self._get_orientations(robust_kp)
        
    
