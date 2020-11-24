import pickle
import numpy as np
import cv2 as cv
import imgfuncs as fcs
import userParam as up
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

######cluster counter
def lineDetect(array):
    linBox = []
    pix = 0
    while pix < len(array):
            if not array[pix]:
                pix+=1
            else:
                wins = []
                gap = 0 
                for j in range(TH_Size):
                    if pix + j < len(array):        
                        if array[pix]:
                            wins.append(pix+j)
                            gap = 0                                
                        else:
                            if gap < GAP_Size:
                                gap += 1
                                wins.append(pix+j)
                            else:                                
                                break
                linBox.append(wins)
                pix+=j
    
    return linBox

import time
################################################################
# a function to detect inteface for one frame using one line test
# this reducaes the line area to 1D to do actual analysis
################################################################
def locateInterface(fileName,r,channels,nlines):
    frame, a = fcs.isolateInterface(fileName,r)
    Boxes = []
    for line in range(nlines):
        chanLine = channels[line]
        x1,y1 = chanLine[0] 
        x2,y2 = chanLine[-1]
        xs = min(x1,x2)
        xl = max(x1,x2)
        area1 = a[y1:y2, xs-chanWidth:xl+chanWidth]
        line1 = []
        line = np.sum(area1,axis=1)/area1.shape[1]
        line2 = np.gradient(line)
        line2 = np.append(line2,0)

        for pix in chanLine:
            x1,y1 = pix
            line1.append(frame[y1][x1].astype(np.int32))
        line3 = np.abs(line1*line2)
        line4 = np.where(line3>Threshold*area1.shape[1],1,0)
        linBox = lineDetect(line4)

        
        Boxes.append(linBox)
    return Boxes
        


### do no edit here edit the userParam file####################
################################################################
def createGlobTime(fileList,r,channels,nlines, i):
    globTime = []
    while i < frameLimit - step:
        print(i)
        frame, a = fcs.isolateInterface(fileList[i],r)
        #frame2, a2 = fcs.isolateInterface(fileList[i+step],r)
        
        dummy = np.copy(frame)
        dummy = cv.merge([dummy,dummy,dummy])
        Boxes = []
        #print("start", i)
        for line in range(nlines):
            pix = 0
            linBox = []
            while pix < len(channels[line]):
                x,y = channels[line][pix]
                #winStart = True
                if frame[y][x] < Threshold:
                    pix+=1
                else:
                    wins = []
                    
                    gap = 0
                    for j in range(TH_Size):
                        #if winStart:
                        #    pix -= 2
                        #winStart = False
                        
                        if pix + j < len(channels[line]):
                            x,y = channels[line][pix+j]        
                            if frame[y][x] > Threshold:
                                wins.append(pix+j)
                                gap = 0                                
                            else:
                                if gap < GAP_Size:
                                    gap += 1
                                    wins.append(pix+j)
                                else:
                                    
                                    break
                    linBox.append(wins)
                    pix+=j

            Boxes.append(linBox)
        

        globTime.append(Boxes)
        i+=step

    pickle.dump(globTime,open("globtime",'wb'))

    return globTime

def createGlobTime2D(fileList,r,channels,nlines, i):
    globTime = []
    while i < frameLimit - step:
        Boxes = []
        frame, a = fcs.isolateInterface(fileList[i],r)
        frame2, a2 = fcs.isolateInterface(fileList[i+step],r)
        thresh = frame
        #print("start", i)
        for line in range(nlines):
            pix = 0
            linBox = []
            for i in range(-chanWidth,chanWidth):
                lineDet =[]
                while pix < len(channels[line]):
                
                    x,y = channels[line][pix]

                    
                    #winStart = True
                    if thresh[y][x] < Threshold:
                        pix+=1
                    else:
                        wins = []
                        
                        gap = 0
                        for j in range(TH_Size):
                            #if winStart:
                            #    pix -= 2
                            #winStart = False
                            
                            if pix + j < len(channels[line]):
                                x,y = channels[line][pix+j]        
                                if thresh[y][x] > Threshold:
                                    wins.append(pix+j)
                                    gap = 0                                
                                else:
                                    if gap < GAP_Size:
                                        gap += 1
                                        wins.append(pix+j)
                                    else:
                                        
                                        break
                        lineDet.append(wins)
                        pix+=j

                linBox.append(lineDet)
            Boxes.append(linBox)

        globTime.append(Boxes)
        i+=step

    pickle.dump(globTime,open("globtime",'wb'))

    return globTime


def locateTraditional(fileName,r,channels,nlines):
    frame, a = fcs.isolateInterface(fileName,r)
    del a
    dummy = np.copy(frame)
    dummy = cv.merge([dummy,dummy,dummy])
    Boxes = []
    #print("start", i)
    for line in range(nlines):
        pix = 0
        linBox = []
        while pix < len(channels[line]):
            x,y = channels[line][pix]
            #winStart = True
            if frame[y][x] < Threshold:
                pix+=1
            else:
                wins = []
                
                gap = 0
                for j in range(TH_Size):
                    #if winStart:
                    #    pix -= 2
                    #winStart = False
                    
                    if pix + j < len(channels[line]):
                        x,y = channels[line][pix+j]        
                        if frame[y][x] > Threshold:
                            wins.append(pix+j)
                            gap = 0                                
                        else:
                            if gap < GAP_Size:
                                gap += 1
                                wins.append(pix+j)
                            else:
                                
                                break
                linBox.append(wins)
                pix+=j

        Boxes.append(linBox)
    
    return np.array(Boxes)