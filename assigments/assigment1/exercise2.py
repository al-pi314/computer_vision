""" general functions """
import sys
from math import ceil

from UZ_utils import *


def inner_variance(lm, lp, rm, rp):
    return lp * rp * (lm - rm)**2

def myhist(I, nbins, min=0, max=255):
    print("myhist using range from {} to {}".format(min, max))
    bins = np.zeros(nbins)
    bins_size = (1 + max - min) / nbins

    data = I.reshape(-1)
    for x in data:
        bin_id = int((x - min) / bins_size)
        bins[bin_id] += 1

    return bins/np.sum(bins)

def plothis(axis, I, nbins):
    axis.bar(np.arange(nbins), myhist(I, nbins))

def plothisadjusted(axis, I, nbins):
    axis.bar(np.arange(nbins), myhist(I, nbins, max=np.max(I), min=np.min(I)))


def otsu(I):
    prob_distribution = myhist(I, 256)
    
    # initial parameters of both groups
    leftProbability = prob_distribution[0]
    rightProbability = 1 - prob_distribution[0]
    leftMean = 0
    rightMean = sum([i * prob_distribution[i] for i in range(1, len(prob_distribution))]) / rightProbability

    # loop through all seperation options
    best_variance = inner_variance(leftMean, leftProbability, rightMean, rightProbability)
    bestSeperation = 0
    blp = leftProbability
    for i in range(1, len(prob_distribution)):
        leftMean = (leftMean * leftProbability) + prob_distribution[i] * i
        leftProbability += prob_distribution[i]
        # ignore empty seperation
        if leftProbability == 0:
            continue
        leftMean /= leftProbability

        rightMean = (rightMean * rightProbability) - prob_distribution[i] * i
        rightProbability -= prob_distribution[i]
        # ignore empty seperation
        if rightProbability == 0:
            continue
        rightMean /= rightProbability

        v = inner_variance(leftMean, leftProbability, rightMean, rightProbability)
        if v >= best_variance:
            best_variance = v
            bestSeperation = i
            blp = leftProbability
    return bestSeperation, blp

""" subexercise A """
def A():
    plt.figure("SUBEXERCISE A")

    I = cv2.imread("./images/bird.jpg", cv2.IMREAD_GRAYSCALE)

    threshold = 80
    columns = 3
    other_thresholds = [i for i in range(20, 240, 10)]
    rows = ceil(len(other_thresholds)/columns)+1

    # Selection Syntax
    SelectionSytnax = I.copy()
    SelectionSytnax[SelectionSytnax < threshold]=0
    SelectionSytnax[SelectionSytnax >= threshold]=1
    ssplt = plt.subplot(rows, columns, 1)
    ssplt.set_title("Selection Syntax")
    ssplt.imshow(SelectionSytnax, cmap="gray")

    # Where Syntax
    WhereSyntax = I.copy()
    WhereSyntax = np.where(WhereSyntax >= threshold, 1, 0)
    ssplt = plt.subplot(rows, columns, 2)
    ssplt.set_title("Where Syntax")
    ssplt.imshow(WhereSyntax, cmap="gray")

    # Try different thresholds
    for th_id in range(len(other_thresholds)):
        p = plt.subplot(rows, columns, 3 + th_id)
        p.imshow(np.where(I.copy() >= other_thresholds[th_id], 1, 0), cmap="gray")

    plt.show()

# thresholds from 60 to 90 are OK

""" subexercise B """
def B():
    plt.figure("SUBEXERCISE B")

    I = cv2.imread("./images/bird.jpg", cv2.IMREAD_GRAYSCALE)

    plt.subplot(2, 2, 1).imshow(I, cmap="gray")
    plothis(plt.subplot(2, 2, 3), I, 100)
    plothis(plt.subplot(2, 2, 4), I, 10) 
    # Normalization is used to not diffirentiate between images with different sizes (number of pixels) but same color distribution
    plt.show()

""" subexercise C """
def C():
    plt.figure("SUBEXERCISE C")

    I = cv2.imread("./images/bird.jpg", cv2.COLOR_BGR2RGB)
    I_GRAYSCALE = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

    plt.subplot(2, 2, 1).imshow(I_GRAYSCALE, cmap="gray")
    plothisadjusted(plt.subplot(2, 2, 3), I_GRAYSCALE, 100)
    plothisadjusted(plt.subplot(2, 2, 4), I_GRAYSCALE, 20)
    plt.show()

""" subexercise D """
def D():
    plt.figure("SUBEXERCISE D")

    CI = cv2.cvtColor(cv2.imread("./images/bright.jpg"),  cv2.COLOR_BGR2GRAY)
    CI = cv2.resize(CI, (0,0), fx=0.5, fy=0.5) 
    plt.subplot(3, 2, 1).imshow(CI, cmap="gray", vmax=255)
    plothis(plt.subplot(3, 2, 3), CI, 10)
    plothis(plt.subplot(3, 2, 4), CI, 100)
    plothisadjusted(plt.subplot(3, 2, 5), CI, 10)
    plothisadjusted(plt.subplot(3, 2, 6), CI, 100)

    plt.show()

""" subexercise E """
def E():

    def addImage(path, PROB_OBJECT, CURR):
        CI = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        th, prob = otsu(CI)
        opencv_th = cv2.threshold(CI, 0, 255, cv2.THRESH_OTSU)[0]
        print("-------- [ image {} ] --------".format(path))
        print("My best seperation:", th)
        print("OpenCV best seperation:", opencv_th)
        print("Agrees with opencv Otsu:", th == opencv_th)
        plt.subplot(SIZE, 2, CURR).imshow(np.where(CI >= th, 1, 0), cmap="gray")
        CURR += 1
        # flip mask
        ABOVE_TH = 1
        UNDER_TH = 0
        print("Object probability {}, expected at least {}".format(prob, PROB_OBJECT))
        if prob <= PROB_OBJECT:
            ABOVE_TH = 0
            UNDER_TH = 1
        plt.subplot(SIZE, 2, CURR).imshow(np.where(CI >= th, ABOVE_TH, UNDER_TH), cmap="gray")
        CURR += 1

        return CURR

    images = ["./images/coins.jpg", "./images/eagle.jpg", "./images/bird.jpg", "./images/custom_dark.jpg", "./images/custom_flash.jpg", "./images/custom_normal.jpg"]
    prob_object = [0.2, 0.5, 0.5, 0.6, 0.6, 0.6]
    SIZE = len(images)
    CURR = 1

    for i in range(len(images)):
        CURR = addImage(images[i], prob_object[i], CURR)

    plt.show()

""" main """
if __name__ == "__main__":
    if len(sys.argv) != 2 or len(sys.argv[1]) != 1:
        print("Please specify subexercise: py exercise.py <letter>")
    else:
        subexercises = {
            "A": A, "B": B, "C": C, "D": D, "E": E
        }
        if subexercises.get(sys.argv[1].upper()) is not None:
            subexercises.get(sys.argv[1])()
