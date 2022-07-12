import numpy as np
import cv2
def removeBG(img):
    height, width = imgo.shape[:2]

    # Create a mask holder
    mask = np.zeros(imgo.shape[:2], np.uint8)

    # Grab Cut the object
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Hard Coding the Rect… The object must lie within this rect.
    rect = (10, 10, width - 30, height - 30)
    cv2.grabCut(imgo, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img1 = imgo * mask[:, :, np.newaxis]
    print(img1)
    # Get the background

    background = imgo - img1

    # Change all pixels in the background that are not black to white
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    # Add the background and the image
    final = background + img1

    # To be done – Smoothening the edges….


    return final

def faceDetectAndFill(final):
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    #
    for (x, y, w, h) in faces:
        final = cv2.rectangle(final, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return final

def skinShow(final):
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    frame = final
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    cv2.imshow("Skinmask",skinMask)
    print(skinMask.shape)
    print(frame.shape)
    skin = cv2.bitwise_not(frame, skinMask)
    # show the skin in the image along with the mask
    #cv2.imshow("images", np.hstack([frame, skin]))
    cv2.imshow("SkinColored",skin)
    cv2.waitKey(0)
    # if the 'q' key is pressed, stop the loop

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Load the Image
imgo = cv2.imread('66245.jpg')
cv2.imshow('Original',imgo)
cv2.waitKey(0)
final=removeBG(imgo)
final=faceDetectAndFill(final)
final=skinShow(final)
cv2.destroyAllWindows()

