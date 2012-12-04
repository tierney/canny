import cv2
import numpy

# Load the image
def CannyImage(image_name):
  img = cv2.imread(image_name + ".jpg")

  # Split out each channel
  blue, green, red = cv2.split(img)

  def medianCanny(img, thresh1, thresh2):
      median = numpy.median(img)
      img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
      return img

  # Run canny edge detection on each channel
  blue_edges = medianCanny(blue, 0.2, 0.3)
  green_edges = medianCanny(green, 0.2, 0.3)
  red_edges = medianCanny(red, 0.2, 0.3)

  # Join edges back into image
  edges = blue_edges | green_edges | red_edges

  # Find the contours
  contours,hierarchy = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions

  # For each contour, find the bounding rectangle and draw it
  for component in zip(contours, hierarchy):
      currentContour = component[0]
      currentHierarchy = component[1]
      x,y,w,h = cv2.boundingRect(currentContour)
      if currentHierarchy[2] < 0:
          # these are the innermost child components
          cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
      elif currentHierarchy[3] < 0:
          # these are the outermost parent components
          cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

  # Finally show the image
  cv2.imwrite(image_name + '-cannied.jpg', img)
  # cv2.imshow('img',img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

CannyImage('maple')
# CannyImage('maple-reduced-0')
CannyImage('maple-reduced-1')
CannyImage('maple-reduced-2')
CannyImage('maple-reduced-3')
CannyImage('maple-reduced-4')
CannyImage('maple-reduced-5')
CannyImage('maple-reduced-6')
CannyImage('maple-reduced-7')
CannyImage('maple-reduced-8')

