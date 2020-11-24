###################################################
# PARAMETERS that are used globally 
# to identify the interfaces
###################################################

#########################################################
# These values are base on some previous human knowledge 
# of channel being analyzed and user estimation
# We can write machine learning code that optimises 
# these automatically but the whole process will be
# considered for the future, 
#######################################################


## use 3,2,50 for our case
## use 8-10,4,170 for brighton
##for other case tune the parameters yourself in a image

TH_Size = 20   # This is max size of interface in pixels
GAP_Size = 15    # usual ~ 20-30% of TH_size
Threshold = 30  # The cutoff the threshold for interfaces
                 #( Make sure there is no noice interference
                 # at the same threshold, in this case set value 
                 # higher as tracking is not possible during noise anyways
                 # )          


## THese tell the code about the geometry of the channel
pollWindow = 1 # what window to check for the tracking
chanWidth = 6 # width of the channel( used in gaussian code
              # and posibily in MM code later)

#these are parameters to display not so important 
trackerheight = 4   
trackerThickness = 1
## still need to add tracking markers code


## step for images ( and how far to process)
step = 1
frameLimit = 40

##pixel to real lenght ratio
pixelRatio = 1