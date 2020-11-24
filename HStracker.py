import numpy as np
import cv2


help_message = '''
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
'''

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(np.int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[::,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*20, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print( help_message)
    try: fn = sys.argv[1]
    except: fn = 0

    import imgfuncs as fcs

    #1.creating a file List
    import fileIOgen as fio
    fileList = fio.retFileList()


    #2.geting region of interest
    r = fcs.getRegion(fileList[0])


    #3.now locate the channels using first image 

    #3.1crop image region and composite
    a1 = cv2.imread(fileList[0],0)
    a1 = fcs.crop(a1,r)
    img = cv2.merge([a1,a1,a1])
    i = 0
    j = 1

    _,prevgray = fcs.isolateInterface(fileList[0],r)
    prev = cv2.merge([prevgray,prevgray,prevgray])
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()
    cv2.namedWindow('flow HSV',2)
    while i<200:
        _,ret = fcs.isolateInterface(fileList[i],r)
        gray = ret#cv2.adaptiveThreshold(ret,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY_INV,11,2)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None,0.5, 3, 15, 3, 5, 1.2, 0)
        i+=1 
        prevgray = gray
        
        cv2.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv2.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)

        ch = 0xFF & cv2.waitKey(0)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print ('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print ('glitch is', ['off', 'on'][show_glitch])
    cv2.destroyAllWindows() 			