#/usr/bin/env python

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

thresholds = {
  0: [0, 0, 0, 0, 0, 0, 0, 0, 5, 95],
  1: [0, 0, 0, 0, 0, 0, 0, 5, 14, 88],
  2: [0, 0, 0, 0, 0, 0, 5, 14, 34, 54],
  3: [0, 0, 0, 0, 0, 5, 14, 34, 34, 14],
  4: [0, 0, 0, 0, 5, 14, 34, 34, 14, 5],
  5: [0, 0, 0, 5, 14, 34, 34, 14, 5, 0],
  6: [0, 0, 5, 14, 34, 34, 14, 5, 0, 0],
  7: [0, 5, 14, 34, 34, 14, 5, 0, 0, 0],
  8: [5, 14, 34, 34, 14, 5, 0, 0, 0, 0],
  9: [14, 34, 34, 14, 5, 0, 0, 0, 0, 0],
  10: [54, 34, 14, 5, 0, 0, 0, 0, 0, 0],
  11: [88, 14, 5, 0, 0, 0, 0, 0, 0, 0],
  12: [95, 5, 0, 0, 0, 0, 0, 0, 0, 0],
  13: [95, 5, 0, 0, 0, 0, 0, 0, 0, 0],
  14: [95, 5, 0, 0, 0, 0, 0, 0, 0, 0],
}

import bisect
import random
import itertools

try:
    xrange
except NameError:
    # Python 3.x
    xrange = range


def weighted_random_choice(seq, weight):
    """Returns a random element from ``seq``. The probability for each element
    ``elem`` in ``seq`` to be selected is weighted by ``weight(elem)``.

    ``seq`` must be an iterable containing more than one element.

    ``weight`` must be a callable accepting one argument, and returning a
    non-negative number. If ``weight(elem)`` is zero, ``elem`` will not be
    considered.

    """
    weights = 0
    elems = []
    for elem in seq:
        w = weight(elem)
        try:
            is_neg = w < 0
        except TypeError:
            raise ValueError("Weight of element '%s' is not a number (%s)" %
                             (elem, w))
        if is_neg:
            raise ValueError("Weight of element '%s' is negative (%s)" %
                             (elem, w))
        if w != 0:
            try:
                weights += w
            except TypeError:
                raise ValueError("Weight of element '%s' is not a number "
                                 "(%s)" % (elem, w))
            elems.append((weights, elem))
    if not elems:
        raise ValueError("Empty sequence")
    ix = bisect.bisect(elems, (random.uniform(0, weights), None))
    return elems[ix][1]

def TwoDDCT(matrix):
  return dct(dct(matrix, axis=0, norm='ortho'), axis=1, norm='ortho')


def TwoDIDCT(matrix):
  return dct(dct(matrix, type=3, axis=0, norm='ortho'), type=3, axis=1, norm='ortho')

def WeightedChoice(h, w):
  hw = h + w
  weighted_choices = zip(epsilons, thresholds[hw])
  population = [val for val, cnt in weighted_choices for i in range(cnt)]
  return random.choice(population)

def Reshift(square):
  for h in range(0, 8):
    for w in range(0, 8):
      if abs(square[h][w]) > 15:
        sign = 1.0 if square[h][w] > 0 else -1.0
        square[h][w] = sign * WeightedChoice(h,w)
  return square


def main(argv):
  image = cv2.imread(argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
  new_image = np.zeros(image.shape)
  height, width = image.shape
  for h in range(0, height, 8):
    for w in range(0, width, 8):
      sub_im = np.zeros((8,8))
      for i in range(h, h+8):
        for j in range(w, w+8):
          sub_im[i-h][j-w] = image[i][j]
      dctd = TwoDDCT(sub_im.astype('float'))

      # Do something here to try to reconstruct the actual values of the matrix
      de_high_freq = Reshift(dctd)

      to_reimplant = np.around(TwoDIDCT(de_high_freq))
      to_reimplant[to_reimplant > 255] = 255
      to_reimplant[to_reimplant < 0] = 0
      # PrintMatrix(to_reimplant)
      for i in range(h, h+8):
        for j in range(w, w+8):
          new_image[i][j] = to_reimplant[i-h][j-w]

  cv2.imwrite(argv[1].strip('.jpg')+'.rev.jpg', new_image)

if __name__=='__main__':
  main(sys.argv)
