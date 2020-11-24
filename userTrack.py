import numpy as np
import copy
import cv2 as cv

import matplotlib.pyplot as plt

#from mpl_toolkits.mplot3d import Axes3D

################################################################
# setting up user parameters
import userParam as up
TH_Size = up.TH_Size
GAP_Size = up.GAP_Size
Threshold = up.Threshold
pollWindow = up.pollWindow
chanWidth = up.chanWidth
trackerheight = up.trackerheight
trackerThickness = up.trackerThickness
step = up.step
frameLimit = 1375#555
pixelRatio = up.pixelRatio
start = 615
### do no edit here edit the userParam file####################
################################################################

def isRectangleOverlap( R1, R2):
    if not R2:
        return False
    
    if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
        return False
    else:
        return True

def rectAreaCMP(R1,R2):
    if not R2:
        return False

    if (abs(R2[2]*R2[3]) >= abs(2*R1[2]*R1[3])) :
        return False
    else:
        return True


### getting our moudules
#getting image funciton
import imgfuncs as fcs

#1.creating a file List
import fileIOgen as fio
fileList = fio.retFileList()

print(len(fileList))

#2.geting region of interest
r = fcs.getRegion(fileList[start])



#3.now locate the channels using first image 

#3.1crop image region and composite
a1 = cv.imread(fileList[start],0)
a1 = fcs.crop(a1,r)
img = cv.merge([a1,a1,a1])

#3.2Allow drawing lines to identify channstepels

cv.namedWindow("LocateInterface", 2)
Box = cv.selectROIs("LocateInterface",img,False,False)



trackerList = []

for win,bbox1 in enumerate(Box):
    p1 = (int(bbox1[0]), int(bbox1[1]))
    p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
    cv.rectangle(img, p1, p2, (255,0,0), 2, 1)
    cv.putText(img, str(win), p1, cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    trackerList.append(tuple(bbox1))
    
cv.namedWindow("Final Image", 2)
cv.imshow("Final Image", img)
    

i = start
j = 1
print('reached loop')
velocity = []
displacement = []
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
from scipy import signal
while i < frameLimit:
    print('in loop val = ', i, fileList[i])
    frame, a = fcs.isolateInterface(fileList[i],r)
    frame2, a2 = fcs.isolateInterface(fileList[i+step],r)
    dummy = cv.merge([a,a,a])
    #Box = globTime[j]
    velBox = []
    newTrackerList =[] 
    for count,window in enumerate(trackerList):
        #tracker = cv.TrackerMedianFlow_create()
        tracker = cv.TrackerCSRT_create()
        ok = tracker.init(a, window)
        if not ok:
            pass
        ok, newWindow = tracker.update(a2)

        org = Box[count]

        arrr  = rectAreaCMP(org, window)

        if ok and arrr:
            bbox1 = window
            p1 = (int(bbox1[0]), int(bbox1[1]))
            p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            b1 = fcs.crop(a, window)
            b2 = fcs.crop(a2,newWindow)
            cv.rectangle(dummy, p1, p2, (255,0,0), 2, 1)
            cv.putText(dummy, str(count), p1, cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            newTrackerList.append(newWindow)
        
        else:
            print("Warning, tracking lost confirm that tracker exists by drawin ta window")
            dummy1 = cv.merge([a2,a2,a2])
            bbox1 = window
            p1 = (int(bbox1[0]), int(bbox1[1]))
            p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            cv.rectangle(dummy1, p1, p2, (255,0,0), 2, 1)
            cv.putText(dummy1, str(count), p1, cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            cv.namedWindow("ReLocateInterface", 2)
            newWindow = cv.selectROI("ReLocateInterface",dummy1,False,False)
            b1 = fcs.crop(a, window)
            b2 = fcs.crop(a2,newWindow)
            newTrackerList.append(newWindow)

        # corr2Darray = normalize(signal.convolve2d(b1,b2))
        # if not corr2Darray.size:
        #     break
        # ny,nx = corr2Darray.shape
        # x = np.linspace(-0.5,0.5,nx)
        # y = np.linspace(-0.5,0.5,ny)    
        # xv,yv = np.meshgrid(x,y)

        # yi,xi = np.unravel_index(corr2Darray.argmax(),corr2Darray.shape)

        old = 0.5*(window[1]+window[3])
        new = 0.5*(newWindow[1]+newWindow[3])
        
        
        
        velBox.append(new)
    
    velocity.append(velBox)
    trackerList = copy.deepcopy(newTrackerList)
    cv.namedWindow("Tracking", 2)
    cv.imshow("Tracking", dummy)
    k = cv.waitKey(1) & 0xFF

    # if k == ord('k') and i>=step:
    #     i-=step
    # elif k == ord('l') and i< len(fileList) - step:
    #     i+=step
    # elif k == ord('q'):
    #     break
    if k == ord('r'):
        reDraw = True
    i+=step
    j+=1
    if i == 1376:
        break
import pickle
#pickle.dump(velocity,open("velocity1",'wb'))
pickle.dump(velocity,open("velocity2",'wb'))