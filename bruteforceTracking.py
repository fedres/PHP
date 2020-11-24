import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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
frameLimit = up.frameLimit

start = 0 #the starting frame  number
### do no edit here edit the userParam file####################
################################################################




### getting our moudules
#getting image funciton
import imgfuncs as fcs

#1.creating a file List
import fileIOgen as fio
fileList = fio.retFileList()


#2.geting region of interest
r = fcs.getRegion(fileList[0])


print(len(fileList))
#3.now locate the channels using first image 

#3.1crop image region and composite
a1 = cv.imread(fileList[0],0)
a1 = fcs.crop(a1,r)
img = cv.merge([a1,a1,a1])

#3.2Allow drawing lines to identify channels
top =[]
bottom = []
offset = 0
def setPoints(event,x,y,flags,param):
    
    global top,bottom
    
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),5,(255,0,0),-1)
        if y > (img.shape[0]/2):
            bottom.append((x-offset,y))
        else:
            top.append((x-offset,y))

cv.namedWindow('Draw Channels', 2)
cv.setMouseCallback('Draw Channels',setPoints)
flag = True
while(flag):
    cv.imshow('Draw Channels',img)
    if top and bottom and len(top) == len(bottom):
        cv.line(img,top[-1],bottom[-1],(0,0,255),1)
    k = cv.waitKey(20) & 0xFF
    
    if k == ord('q'):
        break
    elif k == ord('d'):
        print(top,bottom)
    elif k == ord('a'):
        flag = False  


#3.3using this value show create line
import linefcs as lfcs
channels,nlines = lfcs.createLineList(top,bottom,img)

#4 now lets create a global time index for where we 
#estimate the region of interfaces are located
import createGlobTime as cgtime
globTime = cgtime.createGlobTime(fileList,r,channels,nlines,start)


# Now let us use our brute force algorithm to track the 
# particle  and calculate the velocity
import trackingFunctions as trkfcs


## This is the main code This loop processes images one at a time.
print(len(globTime))

seeTracker = True
cv.namedWindow("vector", 2)
if seeTracker:
    i = start + step
    j = 1  
    while i < frameLimit - 2*step:
        frame, a = fcs.isolateInterface(fileList[i-1],r)
        del a
        dummy = np.copy(frame)
        dummy = cv.merge([dummy,dummy,dummy])
        Boxes = globTime[j-1]
        
        for p in range(nlines):
            cv.line(dummy,top[p],bottom[p],(0,0,255),1)
            #print(Boxes)
        lin =0
        for lb in Boxes:
            for w in lb:
                pix = w[0]
                pixend = w[-1]
                r1 = channels[lin][pix]
                r2 = channels[lin][pixend]
                cv.rectangle(dummy,tuple(np.subtract(r1 , [chanWidth,trackerheight])),tuple(np.add(r2,[chanWidth,trackerheight])),(100,200,200),trackerThickness)
            lin +=1
        cv.imshow('vector',dummy)
        
        k = cv.waitKey(20) & 0xFF

        if k == ord('k') and i>=step:
            i-=step
            j-=1
        elif k == ord('l') and i< len(fileList) - step:
            i+=step
            j+=1
        elif k == ord('q'):
            break 





j = 1
i = start + step
velocity = []
xseries = []
while i < frameLimit - 2*step:
    print("LOOP",j)
    frame1, a1 = fcs.isolateInterface(fileList[i-step],r)
    frame, a = fcs.isolateInterface(fileList[i],r)
    frame2, a2 = fcs.isolateInterface(fileList[i+step],r)
    PreBox = globTime[j-1]
    Box = globTime[j]
    velBox = []
    empty = False
    for ch in range(nlines):
        channel = Box[ch]
        preChannel = PreBox[ch]
        #winPos,winVel = trkfcs.calcVelocityLK(channel,preChannel,channels[ch],frame,frame1,frame2,a,a1,a2)
        winPos,winVel = trkfcs.calcVelocity(channel,preChannel,channels[ch],frame,frame1,frame2)
        velBox.append(winPos)
    velocity.append(velBox)
    xseries.append(velBox)
    i+=step
    j+=1

import pickle
pickle.dump(xseries, open("xseries1",'wb'))
#pickle.dump(xseries, open("xseries",'wb'))
