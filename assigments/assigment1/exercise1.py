""" general functions """
import sys
import time
from math import ceil

import cv2
import numpy as np
from matplotlib import pyplot as plt

from UZ_utils import *


def imgstats(I):
        height, width, channels = I.shape
        dataType = I.dtype
        print("height: {}, width: {}, channels: {}, data type: {}".format(height, width, channels, dataType))

""" subexerices A """
def A():
    plt.figure("SUBEXERCISE A")
    I = imread('./images/umbrellas.jpg')
    imshow(I) # display image
    print(I)  # show image structure

    I = cv2.imread('./images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    imgstats(I)
    imshow(I)

    plt.show()

"""  subexercises B """
def B():
    plt.figure("SUBEXERCISE B")
    I = cv2.imread('./images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    # convert to float dtype and average all channels

    # conversion using for loops
    J = I.copy()
    s = time.time_ns()
    for i in range(len(J)):
        for j in range(len(J[i])):
            avg = sum(J[i][j])/3
            J[i][j] = [avg, avg, avg]
    print(time.time_ns() - s)

    # conversion using np functionalites
    R = I.copy()
    s = time.time_ns()
    R = np.sum(R, axis=2, dtype=np.float64)/3
    R = np.repeat(R[:, :, np.newaxis], 3, axis=2)
    R = R.astype(np.uint8)
    print(time.time_ns() - s)


    # compare results
    print(np.all(R == J))
    imgstats(R)
    imgstats(J)

    # show result
    plt.imshow(R)
    plt.show()


""" subexercise C """
def C():
    # Different color maps present different clues about the images
    # Encoded data in different color maps may differ from what humans percieves
    # Grayscale might be sufficient for data analysis but isn't as usefull for display to humans
    plt.figure("SUBEXERCISE C")
    I = cv2.imread('./images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    images = [
        # correct gray scale image
        {
            "img": I[130:260, 240:450, 1], 
            "cmap":"gray"
        },
        # image with wrong color map interperetation
        {
            "img": I[130:260, 240:450, 1]
        },
        {
            "img": I[130:260, 240:450, 1],
            "cmap": "viridis"
        },
        # gray color map is ingored
        {
            "img": I[130:260, 240:450, :], 
            "cmap":"gray"
        },
        # normal rgb cut out
        {
            "img": I[130:260, 240:450, :], 
        },
        {
            "img": I[0:260, 20:30, :], 
        }
    ]

    rows = ceil(len(images)**0.5)
    for i in range(len(images)):
        plt.subplot(rows, rows, i+1)
        img = images[i]
        if img.get("cmap") is not None:
            plt.imshow(img["img"], cmap=img["cmap"])
        else:
            plt.imshow(img["img"])
    plt.show()

""" subexercise D """
def D():
    plt.figure("SUBEXERCISE D")
    I = cv2.imread('./images/umbrellas.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    T=I.copy()
    # Inverting is the same as substracting current value from max value of uin8 (255)
    T[150:300, 250:450, :] = 255 - T[150:300, 250:450, :]
    plt.imshow(T)
    plt.show()

""" subexercise E"""
def E():
    plt.figure("SUBEXERCISE E")

    P=imread_gray("./images/umbrellas.jpg")
    plt.subplot(1, 3, 1).imshow(P, cmap="gray")
    D = np.uint8(P * 63)
    plt.subplot(1, 3, 2).imshow(D, cmap="gray")
    plt.subplot(1, 3, 3).imshow(D, cmap="gray", vmax=255)
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
