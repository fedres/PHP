import os
## The path where the images to be processed are stored
dirPath = 'F:/images/CuSa_J1_S3_P1/'
#dirPath = '../mosa-bri-j2s5p1/'
#dirPath = '../images/'
#dirPath = '../AnimationGen/images-10/'

def retFileList():
    os.system('ls ' + dirPath + ' > filesList')
    fileList = []
    fileName = 'filesList'
    #fileName = 'mfileList'
    with open(fileName, "r") as f:
        for fileL in f.readlines():
            ifile = dirPath + fileL.rstrip()
            fileList.append(ifile)

    return fileList