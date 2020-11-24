## These are wrapper functions for managing 
# the opencv windows of those images that need displaying 

import cv2 as cv

#wrapper to always draw resizable windows 
def show_(img, name):
    cv.namedWindow(name, 2)
    cv.imshow(name,img)
    return None



