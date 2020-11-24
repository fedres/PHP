import cv2 as cv # this is done to make life easier 
import numpy as np

##This is cropping function used to crop images
#to our required size from the region od interest 
def crop(im, r):
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    return imCrop

#This an reosion dilation filter that is supposed to eliminate
#single pixel noise that are common in low res images
def ero_dil(img,kernel2):
    erosion = cv.erode(img, kernel2, iterations=2)
    final = cv.dilate(erosion, kernel2, iterations=2)
    return final

##THis is a function to isolate interfaces
# it uses a sobel gradient filter (3x3 style)
def isolateInterface(filename, r):
    img = cv.imread(filename,0)
    a = crop(img,r)
    
    #(thresh, a) = cv.threshold(a, 10, 255, cv.THRESH_BINARY)
    kernel = np.array([(1, 0, -1),
                      (2, 0, -2),
                      (1, 0, -1)])


    kernel2 = np.zeros((3, 3), dtype=np.uint8)
    kernel2[1] = np.ones(3, dtype=np.uint8)

    #apply sobel horizontal filter2D  with 0.5 strength
    b = cv.filter2D(a, -1, 0.5*kernel.transpose())
    #gray = cv.adaptiveThreshold(b,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv.THRESH_BINARY_INV,11,2)
    #apply erosion dilation filter2D
    erosion = cv.erode(b, kernel2, iterations=2)
    final = cv.dilate(erosion, kernel2, iterations=2)
    edges = final#cv.Canny(a,100,200)
    return edges,a

def isolateInterfaceImg(img, r):
    
    a = crop(img,r)

    kernel = np.array([(1, 0, -1),
                      (2, 0, -2),
                      (1, 0, -1)])


    kernel2 = np.zeros((3, 3), dtype=np.uint8)
    kernel2[1] = np.ones(3, dtype=np.uint8)

    #apply sobel horizontal filter2D  with 0.5 strength
    b = cv.filter2D(a, -1, 0.5*kernel.transpose())
    #apply erosion dilation filter2D
    erosion = cv.erode(b, kernel2, iterations=2)
    final = cv.dilate(erosion, kernel2, iterations=2)

    return final,a

def isolateInterface2d(filename, r):
    img = cv.imread(filename,0)
    a = crop(img,r)
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    grad_x = cv.Sobel(a, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(a, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    #apply erosion dilation filter2D
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = abs_grad_x#cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad,a


# this is a function to detect vertial lines
# It was added to the Library but not really used
# Its present incase a need arises in the future 
def verticalDetect(img):
    kernel = np.array([(1, 0, -1),
                  (2, 0, -2),
                  (1, 0, -1)])

    img = cv.filter2D(img, -1, kernel)
    return img

def getRegion(filename):
    selROI = cv.imread(filename, 0)
    cv.namedWindow("ROI", 2)
    r = cv.selectROI("ROI",selROI,False,False)

    return r
