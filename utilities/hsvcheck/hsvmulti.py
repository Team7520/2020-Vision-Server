import cv2
import numpy as np
import argparse

def nothing(x):
    pass

def stack_image(image1, image2):
    # numpy_horizontal = np.hstack((image1, image2))

    numpy_horizontal_concat = np.concatenate((image1, image2), axis=1)    
    # cv2.imshow('image', numpy_horizontal_concat)
    return numpy_horizontal_concat

def stack_images(image1, image2, image3, image4):
    # numpy_vertical = np.vstack((image, grey_3_channel))
    # numpy_horizontal = np.hstack((image, grey_3_channel))

    # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
    # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

    cat1 = np.concatenate((image1, image2), axis=1)  
    cat2 = np.concatenate((image3, image4), axis=1)    
    catf = np.concatenate((cat1, cat2), axis=0)
    cv2.imshow('image', catf)
    return catf

def hsv_image(image, lower, upper):
    blurred = cv2.GaussianBlur(image, (27, 27), 0)
    median = cv2.medianBlur(blurred, 15)
    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(image,image,mask = mask)   
    return result

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('file', type=argparse.FileType('r'), nargs='+')

args = vars(ap.parse_args())

# Create a window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 832,900)
# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMin', 'image', 20)
cv2.setTrackbarPos('SMin', 'image', 100)
cv2.setTrackbarPos('VMin', 'image', 75)
cv2.setTrackbarPos('HMax', 'image', 100)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Load image
images = []
print(args["file"])

for file in args["file"]:
    images.append(cv2.imread(file.name))
    
while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    imagerows = []
    firstCol = None
    for i in range(len(images)):
        result = hsv_image(images[i], lower, upper)
        result = np.concatenate((images[i], result), axis=1)  
        if firstCol is None:
            firstCol= result
        else:
            firstCol = np.concatenate((firstCol, result), axis=1)  
            imagerows.append(firstCol)
            firstCol = None
    
    if not firstCol is None:
        imagerows.append(firstCol)

    finalimage = None
    for i in range(len(imagerows)):
        if finalimage is None:
            finalimage= imagerows[i]
        else:
            finalimage = np.concatenate((finalimage, imagerows[i]), axis=0)  

    # Display result image    
    cv2.imshow('image', finalimage)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()