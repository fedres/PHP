import numpy as np
import cv2 as cv
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
pixelRatio = up.pixelRatio


### do no edit here edit the userParam file####################
################################################################
def channelMatcher(channel,preChannel,frame2,line):
    empty = False
    if len(preChannel) == len(channel):
            print("All good")
    else:
        print("Not tracking or new one formed or lost")
        size = min(len(preChannel),len(channel))
        if size == 0:
            print("Some list is empty")
            if not channel:
                empty = True
                for interface in preChannel:
                    channel.append(interface)
                return channel,preChannel,empty
                
                
        form = 0
        if len(preChannel) == size:
            print("New formed")
            form = 1
        else:
            print("Something lost")
            form = -1
        while (not len(channel) == len(preChannel)) and (not empty):
            wloc = -1
            for w in range(size):
                print("Interface",w)
                preInter = preChannel[w]
                inter = channel[w]
                delta = inter[0] - preInter[0]
                if abs(delta) > pollWindow*step:   
                    print("WARNING: not the same interface", delta)
                    wloc = w
                    break
                else:
                    print("This interface looks good") 
            if form == 1 and not wloc== -1:
                channel.pop(wloc)
            elif form ==-1 and not wloc== -1:
                mask = []
                for pix in preChannel[wloc]:
                    x1,y1 = line[pix]
                    mask.append(frame2[y1][x1])
                mask = np.array(mask)
                if mask.any() > 0:
                    channel.append(preChannel[wloc])
            elif form == 1 and wloc == -1:
                channel.pop()
                print("Removing bad(new that we dont care about) interface in the end")
            elif form == -1 and wloc == -1:
                mask = []
                for pix in preChannel[wloc]:
                    x1,y1 = line[pix]
                    mask.append(frame2[y1][x1])
                mask = np.array(mask)
                if mask.any() > 0:
                    channel.append(preChannel[wloc])
            
            size = min(len(preChannel),len(channel))
    

    return channel,preChannel,empty

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
#function to perform normed convolution
#to get pdf and evaluate the most probable value

def two_vel(t11,t12,pollRange):
    
    t12 = np.array(t12)
    t11 = np.array(t11)
    
    #corrList1 = normalize(cv.matchTemplate(t12,t11,cv.TM_CCORR_NORMED))
    corrList1 = normalize(np.correlate(t12,t11,'same'))
    x = np.linspace(-1,1,len(corrList1))
    #corrList1 = np.where(corrList1 > 0.7,corrList1,0)
    freq = x.dot(corrList1)
    
    if np.sum(corrList1) == 0.0:
        return 0

    probVal = freq/np.sum(corrList1)
    #probVal = x[np.argmax(corrList1)]

    return probVal

def two_vel2D(t11,t12,pollRange):
    flat_image = t11
    r, c = np.shape(flat_image)
    gd = 2

    flow = cv.calcOpticalFlowFarneback(t11,t12, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    pag = flow[...,1]
    pagy = flow[::gd,::gd,1]
    pagx = flow[::gd,::gd,0]
    
    y = np.sum(pag,axis=1)/pag.shape[1]
    vel = np.mean(y)

    test_slice = flat_image[::gd,::gd]  # sampling

    fig,ax=plt.subplots(1,1)
    the_image = ax.imshow(
                    flat_image,
                    zorder=0,alpha=1.0,
                    cmap="Greys_r",
                    origin="upper",
                    interpolation="hermite",
                )
    plt.colorbar(the_image)            
    Y, X = np.mgrid[0:r:gd, 0:c:gd]
    ax.quiver(X, Y, pagx, pagy, color='r')


    plt.savefig("mygraph.png")
    cv.namedWindow('quiver', 2)
    img  = cv.imread('mygraph.png')
    cv.imshow('quiver',img)
    k = cv.waitKey(0) & 0xFF
    
    del fig,ax
    return  vel
#frame 1 is previous time and frame 2 is next time
# frame is the current time 
def calcVelocity(channel,preChannel,line,frame,frame1,frame2):
    winPos = []
    winVel = []
    channel,preChannel,empty = channelMatcher(channel,preChannel,frame2,line)
    # if empty:
    #     winVel.append(float("NaN"))
    #     winPos.append(float("NaN"))
    #     return winPos,winVel
    if empty:
        for w in range(len(channel)):
            winPos.append(float("NaN"))
            winVel.append(float("NaN"))
        return winPos,winVel

    for w in range(len(channel)):
            
        preInter = preChannel[w]
        inter = channel[w]
        size = len(preInter)
        t11 = []
        t12 = []
        pix1 = preInter[0]-pollWindow
        pix = preInter[0]

        for ws in range(size):
            x1,y1 = line[pix+ws]
            t11.append(frame1[y1][x1])

        for ws in range(size+2*pollWindow):
            if pix1+ws < len(line):

                x1,y1 = line[pix1+ws]
                t12.append(frame[y1][x1])
            else:
                t12.append(np.uint8(0))

        curVel = two_vel(t11, t12, pollWindow)

        winPos.append( pix + np.argmax(t11) + curVel)
        winVel.append(curVel*pixelRatio)
    
    return winPos,winVel

def calcVelocityLK(channel,preChannel,line,frame,frame1,frame2,a,a1,a2):
    winPos = []
    winVel = []
    channel,preChannel,empty = channelMatcher(channel,preChannel,frame2,line)

    if empty:
        for w in range(len(channel)):
            winPos.append(float("NaN"))
            winVel.append(float("NaN"))
        return winPos,winVel

    for preInter in channel:
            
        
        
        t11 = []
        t12 = []
    
        start = preInter[0]
        end = preInter[-1]
        x1,y1 =  line[start]
        x2,y2 =  line[end]
        xs = min(x1,x2)
        xl = max(x1,x2)
        # R = (xs-chanWidth,y1-trackerheight,xl+2*chanWidth-xs,y2+2*trackerheight-y1)
        # print(R)
        t11 = a[y1-pollWindow:y2,xs-chanWidth:xl+chanWidth]
        t12 = a2[y1-pollWindow:y2,xs-chanWidth:xl+chanWidth]
        
        curVel = two_vel2D(t11, t12, pollWindow)

        winPos.append( start + np.argmax(t11))
        winVel.append(curVel*pixelRatio)
    
    return winPos,winVel