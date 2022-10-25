

import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

import exercise2 as ex2

""" general """
def square_SE(n):
    return np.ones((n,n), np.uint8)

def close(I, se):
    I_d = cv2.dilate(I, se)
    return cv2.erode(I_d, se)

def open(I, se):
    I_e = cv2.erode(I, se)
    return cv2.dilate(I_e, se)

def toThreeChannles(MASK):
    return np.repeat(np.expand_dims(MASK, 2), 3, axis=2)

def immask(I, MASK):
        EXP_MASK = toThreeChannles(MASK)
        return I * EXP_MASK

""" subexercise A """
def A():
    plt.figure("SUBEXERCISE A")

    I = cv2.imread("./images/mask.png", cv2.IMREAD_GRAYSCALE)

    N_MAX = 10
    N_STEP = 2
    columns = ["dilate", "erode", "open (errode & dilate)", "close (dilate & erode)", "open & close", "close & open"]
    fig, axes = plt.subplots(nrows=int(N_MAX/N_STEP), ncols=len(columns), figsize=(12, 8))

    for j in range(len(axes[0])):
        axes[0, j].set_title(columns[j])
    
    for i in range(len(axes[:, 0])):
        axes[i, 0].set_ylabel("SE size {}".format(1 + i * N_STEP), size='large')

    for n in range(1, N_MAX+1, N_STEP):
        se = square_SE(n)
        axes[int((n-1) / N_STEP), 0].imshow(cv2.dilate(I, se), cmap="gray")
        axes[int((n-1) / N_STEP), 1].imshow(cv2.erode(I, se), cmap="gray")
        axes[int((n-1) / N_STEP), 2].imshow(open(I, se), cmap="gray")
        axes[int((n-1) / N_STEP), 3].imshow(close(I, se), cmap="gray")
        axes[int((n-1) / N_STEP), 4].imshow(close(open(I, se), se), cmap="gray")
        axes[int((n-1) / N_STEP), 5].imshow(open(close(I, se), se), cmap="gray")
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    plt.show()

""" subexercise B """
def B():
    plt.figure("SUBEXERCISE B")

    BIRD = cv2.imread("./images/bird.jpg", cv2.IMREAD_GRAYSCALE)
    plt.subplot(2, 4, 1).imshow(BIRD, cmap="gray")


    BIRD = np.uint8(np.where(BIRD >= 52, 1, 0))
    plt.subplot(2, 4, 2).imshow(BIRD, cmap="gray")

    functions = [close, cv2.dilate, open, close, cv2.erode]
    se = [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), cv2.getStructuringElement(cv2.MORPH_CROSS, (12, 12)), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))]

    BETTER_BIRD = BIRD.copy()
    for i in range(len(functions)):
        BETTER_BIRD = functions[i](BETTER_BIRD, se[i])
        plt.subplot(2, 4, 3 + i).imshow(BETTER_BIRD, cmap="gray")
    plt.show()

    return BETTER_BIRD

""" subexercise C """
def C():
    BETTER_BIRD = B()

    ORIGINAL = cv2.cvtColor(cv2.imread("./images/bird.jpg"), cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1).imshow(ORIGINAL)
    plt.subplot(1, 3, 2).imshow(BETTER_BIRD, cmap="gray")
    plt.subplot(1, 3, 3).imshow(immask(ORIGINAL, BETTER_BIRD))
    plt.show()

""" subexercise D """
def D():
    plt.figure("SUBEXERCISE D")

    EAGLE = cv2.cvtColor(cv2.imread("./images/eagle.jpg"), cv2.COLOR_BGR2RGB)
    EAGLE_GRAYSCALE = cv2.cvtColor(EAGLE, cv2.COLOR_RGB2GRAY)
    ABOVE_TH = 1
    UNDER_TH = 0

    MASK_TH, PROB = ex2.otsu(EAGLE_GRAYSCALE)
    MASK = np.uint8(np.where(EAGLE_GRAYSCALE >= MASK_TH, ABOVE_TH, UNDER_TH))
    MASK = open(MASK, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    MASK = close(MASK, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))

    plt.subplot(2, 3, 1).imshow(EAGLE)
    plt.subplot(2, 3, 2).imshow(MASK, cmap="gray")
    plt.subplot(2, 3, 3).imshow(immask(EAGLE, MASK))

    # Because the object has darker tones
    # If we assume that the object represents smaller poriton of the image we could test to see if the gaussian distribution has a higher probability in bright or dark pixels
    # We can also adjust the bottom condition to match assumed portion of the screen that background occupies

    # dark objects are on the left of TH, light objects are on the right of TH
    # function returns probability of a pixel being on the left
    # we assume that most pixels belong to the background so 
    # if the probability of a pixel being on the left is very low that means that out object is light and background is dark
    # if the probability of a pixel being on the left is very high that means that the background is dark and our object is light

    # invert mask
    print(PROB)
    if PROB <= 0.5:
        ABOVE_TH = 0
        UNDER_TH = 1

    MASK = np.uint8(np.where(EAGLE_GRAYSCALE >= MASK_TH, ABOVE_TH, UNDER_TH))
    MASK = close(MASK, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    MASK = open(MASK, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))

    plt.subplot(2, 3, 4).imshow(EAGLE)
    plt.subplot(2, 3, 5).imshow(MASK, cmap="gray")
    plt.subplot(2, 3, 6).imshow(immask(EAGLE, MASK))
    plt.show()

    plt.show()

""" subexercise E """
def E():
    plt.figure("SUBEXERCISE E")

    COINS = cv2.cvtColor(cv2.imread("./images/coins.jpg"), cv2.COLOR_BGR2RGB)
    COINS_GRAYSCALE = cv2.cvtColor(COINS, cv2.COLOR_RGB2GRAY)

    MASK_TH, PROB = ex2.otsu(COINS_GRAYSCALE)
    ABOVE_TH = 1
    UNDER_TH = 0
    # invert mask
    if PROB <= 0.5:
        ABOVE_TH = 0
        UNDER_TH = 1
    thcv2 = cv2.threshold(COINS_GRAYSCALE, 0, 255, cv2.THRESH_OTSU)
    print(MASK_TH, thcv2[0])

    MASK = np.uint8(np.where(COINS_GRAYSCALE >= MASK_TH, ABOVE_TH, UNDER_TH))
    MASK = close(MASK, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)))

    plt.subplot(2, 3, 1).imshow(COINS)
    plt.subplot(2, 3, 2).imshow(MASK, cmap="gray")
    plt.subplot(2, 3, 3).imshow(immask(COINS, MASK))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(MASK)
    
    # adjust coins mask by removing components coresponding to small coins
    BIG_COINS_MASK = np.ones_like(COINS)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 700:
            BIG_COINS_MASK[labels == i] = 0
    BIG_COINS_MASK = cv2.erode(BIG_COINS_MASK, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    SMALL_COINS = COINS.copy()
    SMALL_COINS[BIG_COINS_MASK == 0] = 255

    plt.subplot(2, 3, 4).imshow(COINS)
    plt.subplot(2, 3, 5).imshow(BIG_COINS_MASK * 255, cmap="gray")
    plt.subplot(2, 3, 6).imshow(SMALL_COINS)

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
