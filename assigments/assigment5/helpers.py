from random import sample

import cv2
import numpy as np


def gaussdx(sigma):
    f = lambda x: np.exp(-x**2/(2*sigma**2)) * (-x/(np.sqrt(2*np.pi) * sigma**3))
    r = np.array(list(map(f, np.arange(-3*sigma, 3*sigma+1))))
    return (r / sum(abs(r))).reshape(1, r.shape[0])

def gaussian(sigma):
    f = lambda x: np.exp(-x**2/(2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
    r = np.array(list(map(f, np.arange(-3*sigma, 3*sigma+1))))
    return (r / sum(r)).reshape(1, r.shape[0])

def g_gdx(o=2):
    if o is None:
        o = 2
    return gaussian(o), -gaussdx(o)

def imgDX(I, o=2):
    g, gdx = g_gdx(o)
    return cv2.filter2D(cv2.filter2D(I, -1, kernel=g.T), -1, kernel=gdx)

def imgDY(I, o=2):
    g, gdx = g_gdx(o)
    return cv2.filter2D(cv2.filter2D(I, -1, kernel=g), -1, kernel=gdx.T)


def euclidian(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def inliers(pointsA, pointsB, H, t):
    points_idx = []
    perror = 0
    for i in range(len(pointsA)):
        # point: [y, x] --> [x, y, 1]
        v = np.matmul(H, [pointsA[i][1], pointsA[i][0], 1]) 
        v = v / v[-1]

        err = euclidian(v[0], v[1], pointsB[i][1], pointsB[i][0])
        perror += err
        if err < t:
            points_idx.append(i)
    return points_idx, perror / len(pointsA)

def estimate_homography(pointsA, pointsB):
    # construct matrix A of homogenous system Ah = 0
    A = np.zeros((2*len(pointsA), 9))
    for i in range(len(pointsA)):
        yA, xA = pointsA[i]
        yB, xB = pointsB[i]

        A[2*i] = [xA, yA, 1, 0, 0, 0, -xB*xA, -xB*yA, -xB]
        A[2*i +1] = [0, 0, 0, xA, yA, 1, -yB*xA, -yB*yA, -yB]

    # preform SVD on A
    _, _, VT = np.linalg.svd(A)

    # compute h
    h = VT[-1, :] / VT[-1, -1]

    # reshape h to 3x3 matrix
    H = h.reshape((3, 3))
    return H  

def ransac(pointsA, pointsB, k=4, samples=4, t=0.001, p=0.99, e=0.1, expected_err=5, estimatior=estimate_homography, inliers_func=inliers):
    # subtask C*
    if k is None:
        # 1 - p = (1 - (1 - e)**samples)**k
        # (1 - e)**samples --> probability of all samples being inliers
        # 1 - (1 - e)**samples --> probability of at least one sample being an outlier
        # (1 - (1 - e)**samples)**k --> probability of all k samples having at least one outlier
        # 1 - p --> probability of drawing k samples with all of them having at least one outlier
        k = np.ceil(np.log(1 - p) / np.log(1 - (1 - e)**samples)).astype(int)
        print(f"Automatically detetmined k = {k} using p = {p}, e = {e} and samples = {samples}")

    options = range(len(pointsA))
    k_match_sets = set()
    while len(k_match_sets) < k:
        k_match_sets.add(tuple(sample(options, samples)))

    best_inliers = []
    best_H = None
    best_perror = None
    for matches in k_match_sets:
        H = estimatior(np.take(pointsA, matches, axis=0), np.take(pointsB, matches, axis=0))

        inliers_idx, _ = inliers_func(pointsA, pointsB, H, t)
        if len(inliers_idx) > len(pointsA) * 0.5:
            H = estimatior(np.take(pointsA, inliers_idx, axis=0), np.take(pointsB, inliers_idx, axis=0))
            _, perror = inliers_func(pointsA, pointsB, H, t)
            if best_perror is None or perror < best_perror:
                best_perror = perror
                best_H = H
                best_inliers = inliers_idx
                # subtask C*
                # add break condition if a sufficient match is found
                if best_perror < expected_err:
                    break
    return best_H, best_inliers, best_perror

def simple_descriptors_wrapper(I, Y, X):
    points = np.array((Y, X)).T
    return simple_descriptors(I, Y, X), points

def hellinger_distance(hist_A, hist_B):
    return np.sum((np.sqrt(hist_A) - np.sqrt(hist_B))**2) ** 0.5 / np.sqrt(2)

def find_correspondences(descriptorsA, descriptorsB, reverse=False):
    matches = []
    distances = []
    for dAi in range(len(descriptorsA)):
        # default basepoint for comparison
        if not reverse:
            matches.append((dAi, 0))
        else:
            matches.append((0, dAi))
        distances.append(hellinger_distance(descriptorsA[dAi], descriptorsB[0]))

        # find best match for point from A
        for dBi in range(1, len(descriptorsB)):
            distance = hellinger_distance(descriptorsA[dAi], descriptorsB[dBi])
            if distance < distances[-1]:
                if not reverse:
                    matches[-1] = (dAi, dBi)
                else:
                    matches[-1] = (dBi, dAi)
                distances[-1] = distance
    return matches

def nonmaxsuppression(I, nb=1):
    if nb == 0:
        return I

    Is = I.copy()
    box_step = 2*nb
    for y in range(0, Is.shape[0], box_step):
        for x in range(0, Is.shape[1], box_step):
            y_lim = min(y + box_step, Is.shape[0] -1)
            x_lim = min(x + box_step, Is.shape[1] -1)
            neighboruhood = Is[y:y_lim, x:x_lim]
            maxiumum = np.max(neighboruhood)
            maximums = np.argwhere(neighboruhood == maxiumum)
            maximum_index = maximums[len(maximums)//2]
            Is[y:y_lim, x:x_lim] = 0
            Is[y+maximum_index[0], x+maximum_index[1]] = maxiumum
            
    return Is

def harris_points(I, o=2, o_after=None, alpha=0.06, t=None, nb=0):
    if o_after is None:
        o_after = 1.6 * o
    afterGK = np.flip(gauss(o_after))
    Ix = imgDX(I, o)
    Iy = imgDY(I, o)
    
    C11 = cv2.filter2D(Ix**2, ddepth=-1, kernel=afterGK)
    C12 = cv2.filter2D(Ix*Iy, ddepth=-1, kernel=afterGK)
    C21 = cv2.filter2D(Iy*Ix, ddepth=-1, kernel=afterGK)
    C22 = cv2.filter2D(Iy**2, ddepth=-1, kernel=afterGK)

    detC = C11*C22 - C12*C21
    traceC = C11 + C22

    value = detC - alpha * (traceC**2)

    if t is not None:
        value[value < t] = 0
        value = nonmaxsuppression(value, nb)
    return value

def find_matches(IA, IB, o=6, t=10**-6, nb=10, descriptors=simple_descriptors_wrapper):
    keypoints_A = np.argwhere(harris_points(IA, o=o, t=t, nb=nb) > 0)
    keypoints_B = np.argwhere(harris_points(IB, o=o, t=t, nb=nb) > 0)

    descriptorsA, keypoints_A = descriptors(IA, keypoints_A[:, 0], keypoints_A[:, 1])
    descriptorsB, keypoints_B = descriptors(IB, keypoints_B[:, 0], keypoints_B[:, 1])

    correspondences_A_B = find_correspondences(descriptorsA, descriptorsB)
    correspondences_B_A = find_correspondences(descriptorsB, descriptorsA, reverse=True)

    symetric_correspondences = set(correspondences_A_B).intersection(set(correspondences_B_A))

    correspondences_info = list(zip(*symetric_correspondences))

    points_A_ordered = np.take(keypoints_A, correspondences_info[0], axis=0)
    points_B_ordered = np.take(keypoints_B, correspondences_info[1], axis=0)

    return points_A_ordered, points_B_ordered

def simple_descriptors(I, Y, X, n_bins = 16, radius = 40, sigma = 2):
	"""
	Computes descriptors for locations given in X and Y.

	I: Image in grayscale.
	Y: list of Y coordinates of locations. (Y: index of row from top to bottom)
	X: list of X coordinates of locations. (X: index of column from left to right)

	Returns: tensor of shape (len(X), n_bins^2), so for each point a feature of length n_bins^2.
	"""

	assert np.max(I) <= 1, "Image needs to be in range [0, 1]"
	assert I.dtype == np.float64, "Image needs to be in np.float64"

	g = gauss(sigma)
	d = gaussdx(sigma)

	Ix = convolve(I, g.T, d)
	Iy = convolve(I, g, d.T)
	Ixx = convolve(Ix, g.T, d)
	Iyy = convolve(Iy, g, d.T)

	mag = np.sqrt(Ix ** 2 + Iy ** 2)
	mag = np.floor(mag * ((n_bins - 1) / np.max(mag)))

	feat = Ixx + Iyy
	feat += abs(np.min(feat))
	feat = np.floor(feat * ((n_bins - 1) / np.max(feat)))

	desc = []

	for y, x in zip(Y, X):
		miny = max(y - radius, 0)
		maxy = min(y + radius, I.shape[0])
		minx = max(x - radius, 0)
		maxx = min(x + radius, I.shape[1])
		r1 = mag[miny:maxy, minx:maxx].reshape(-1)
		r2 = feat[miny:maxy, minx:maxx].reshape(-1)

		a = np.zeros((n_bins, n_bins))
		for m, l in zip(r1, r2):
			a[int(m), int(l)] += 1

		a = a.reshape(-1)
		a /= np.sum(a)

		desc.append(a)

	return np.array(desc)



def gauss(sigma):
	x = np.arange(np.floor(-3 * sigma), np.ceil(3 * sigma + 1))
	k = np.exp(-(x ** 2) / (2 * sigma ** 2))
	k = k / np.sum(k)
	return np.expand_dims(k, 0)


def gaussdx(sigma):
	x = np.arange(np.floor(-3 * sigma), np.ceil(3 * sigma + 1))
	k = -x * np.exp(-(x ** 2) / (2 * sigma ** 2))
	k /= np.sum(np.abs(k))
	return np.expand_dims(k, 0)


def convolve(I: np.ndarray, *ks):
	"""
	Convolves input image I with all given kernels.

	:param I: Image, should be of type float64 and scaled from 0 to 1.
	:param ks: 2D Kernels
	:return: Image convolved with all kernels.
	"""
	for k in ks:
		k = np.flip(k)  # filter2D performs correlation, so flipping is necessary
		I = cv2.filter2D(I, cv2.CV_64F, k)
	return I
def simple_descriptors_wrapper(I, Y, X):
    points = np.array((Y, X)).T
    return simple_descriptors(I, Y, X), points

def find_matches(IA, IB, o=6, t=10**-6, nb=10, descriptors=simple_descriptors_wrapper):
    keypoints_A = np.argwhere(harris_points(IA, o=o, t=t, nb=nb) > 0)
    keypoints_B = np.argwhere(harris_points(IB, o=o, t=t, nb=nb) > 0)

    descriptorsA, keypoints_A = descriptors(IA, keypoints_A[:, 0], keypoints_A[:, 1])
    descriptorsB, keypoints_B = descriptors(IB, keypoints_B[:, 0], keypoints_B[:, 1])

    correspondences_A_B = find_correspondences(descriptorsA, descriptorsB)
    correspondences_B_A = find_correspondences(descriptorsB, descriptorsA, reverse=True)

    symetric_correspondences = set(correspondences_A_B).intersection(set(correspondences_B_A))

    correspondences_info = list(zip(*symetric_correspondences))

    points_A_ordered = np.take(keypoints_A, correspondences_info[0], axis=0)
    points_B_ordered = np.take(keypoints_B, correspondences_info[1], axis=0)

    return points_A_ordered, points_B_ordered