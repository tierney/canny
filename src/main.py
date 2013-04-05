#!/usr/bin/env python

import cv
import cv2
import sys
from scipy.fftpack import dct
import numpy as np

epsilons = [0.99074126327462231,
            3.6661674303584841,
            9.3952604455155004,
            18.760113220064738,
            37.738436922521913,
            76.529427414425342,
            161.6914905923812,
            511.71345539866354,
            766.65549568965514,
            1258.5326821590593]

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
  total_excess = []
  for h in range(8):
    for w in range(8):
      if h == 0 and w == 0:
        continue
      total_excess.append(abs(matrix[h][w]) - level)
      if abs(matrix[h][w]) > level:
        matrix[h][w] = level # if matrix[h][w] > level else -level

  return matrix, total_excess

def main(argv):
  image = cv2.imread('../data/maple.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

  for level in range(5, 40, 5):
    excess = []
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

        de_high_freq, avg_excess = RemoveHighFreqCoefficients(dctd, level)

        excess.append(avg_excess)

        to_reimplant = np.around(TwoDIDCT(de_high_freq))
        to_reimplant[to_reimplant > 255] = 255
        to_reimplant[to_reimplant < 0] = 0
        # PrintMatrix(to_reimplant)
        for i in range(h, h+8):
          for j in range(w, w+8):
            new_image[i][j] = to_reimplant[i-h][j-w]

    cv2.imwrite('maple-reduced-%d.jpg' % level, new_image)
    print [np.std(exc) for exc in excess]
    with open('data.csv','a') as fh:
      for exc in excess:
        for ex in exc:
          if float(ex) > 0:
            fh.write('%f\n' % ex)

if __name__=='__main__':
  main(sys.argv)
