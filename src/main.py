#!/usr/bin/env python

import cv
import cv2
import sys
from scipy.fftpack import dct
import numpy as np

def TwoDDCT(matrix):
  return dct(dct(matrix, axis=0, norm='ortho'), axis=1, norm='ortho')


def TwoDIDCT(matrix):
  return dct(dct(matrix, type=3, axis=0, norm='ortho'), type=3, axis=1, norm='ortho')

def GetImage():
  pass

def PrintMatrix(matrix):
  for h in range(8):
    for w in range(8):
      print matrix[h][w],
    print

def RemoveHighFreqCoefficients(matrix, level):
  # level in the range 7 down to 1. 1 is the most severe trimming of the DCT
  # coefficients.
  for h in range(8):
    for w in range(8):
      if w >= level - h:
        matrix[h][w] = 0
  return matrix

def main(argv):
  image = cv2.imread('maple.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

  for level in range(9):
    new_image = np.zeros(image.shape)
    height, width = image.shape
    for h in range(0, height, 8):
      for w in range(0, width, 8):
        sub_im = np.zeros((8,8))
        for i in range(h, h+8):
          for j in range(w, w+8):
            sub_im[i-h][j-w] = image[i][j]
        # PrintMatrix(sub_im)
        dctd = TwoDDCT(sub_im.astype('float'))

        de_high_freq = RemoveHighFreqCoefficients(dctd, level)
        to_reimplant = np.around(TwoDIDCT(de_high_freq))
        to_reimplant[to_reimplant > 255] = 255
        to_reimplant[to_reimplant < 0] = 0
        # PrintMatrix(to_reimplant)
        for i in range(h, h+8):
          for j in range(w, w+8):
            new_image[i][j] = to_reimplant[i-h][j-w]

    cv2.imwrite('maple-reduced-%d.jpg' % level, new_image)

if __name__=='__main__':
  main(sys.argv)
