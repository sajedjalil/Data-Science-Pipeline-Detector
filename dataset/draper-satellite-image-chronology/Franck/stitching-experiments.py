########################################################################
#
# Taking the1owls Image Matching script and experimenting;
# - downsizing images to speed up
#
# Want to extract the warp parameters
#
########################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

print(cv2.__version__)


# def im_stitcher(imp1, imp2, pcntDownsize = 1.0, withTransparency=False):
    
#     #Read image1
#     image1 = cv2.imread(imp1)
    
#     # perform the resizing of the image by pcntDownsize and create a Grayscale version
#     dim1 = (int(image1.shape[1] * pcntDownsize), int(image1.shape[0] * pcntDownsize))
#     img1 = cv2.resize(image1, dim1, interpolation = cv2.INTER_AREA)
#     img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
#     #Read image2
#     image2 = cv2.imread(imp2)
    
#     # perform the resizing of the image by pcntDownsize and create a Grayscale version
#     dim2 = (int(image2.shape[1] * pcntDownsize), int(image2.shape[0] * pcntDownsize))
#     img2 = cv2.resize(image2, dim2, interpolation = cv2.INTER_AREA)
#     img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
#     #use BRISK to create keypoints in each image
#     brisk = cv2.BRISK_create()
#     kp1, des1 = brisk.detectAndCompute(img1Gray,None)
#     kp2, des2 = brisk.detectAndCompute(img2Gray,None)
    
#     # use BruteForce algorithm to detect matches among image keypoints 
#     dm = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    
#     matches = dm.knnMatch(des1,des2, 2)
#     matches_ = []
#     for m in matches:
#         if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
#             matches_.append((m[0].trainIdx, m[0].queryIdx))
    
#     kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1,1,2)
#     kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1,1,2)
    
    
#     H, mask = cv2.findHomography(kp2_,kp1_, cv2.RANSAC, 4.0)
#     h1,w1 = img1.shape[:2]
#     h2,w2 = img2.shape[:2]
    
#     pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
#     pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    
#     pts2_ = cv2.perspectiveTransform(pts2, H)
#     pts = np.concatenate((pts1, pts2_), axis=0)
    
#     [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
#     [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
#     t = [-xmin,-ymin]
    
#     Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    
#     #warp the colour version of image2
#     im = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
#     #overlay colur version of image1 to warped image2
#     if withTransparency == True:
#         h3,w3 = im.shape[:2]
#         bim = np.zeros((h3,w3,3), np.uint8)
#         bim[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
        
#         #imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         #imColor = cv2.applyColorMap(imGray, cv2.COLORMAP_JET)
        
#         #im =(im[:,:,2] - bim[:,:,2])
#         im = cv2.addWeighted(im,0.6,bim,0.6,0)
#     else:
#         im[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
#     return(im)

# ##########################################################
# #
# # Match all combinations of one set of images
# #
# ##########################################################

# #img104_1_166_1 = im_stitcher("../input/test_sm/set104_1.jpeg", "../input/test_sm/set166_1.jpeg", 0.4, True)
# #img104_2_166_4 = im_stitcher("../input/test_sm/set104_2.jpeg", "../input/test_sm/set166_4.jpeg", 0.4, True)
# #img104_3_166_5 = im_stitcher("../input/test_sm/set104_3.jpeg", "../input/test_sm/set166_5.jpeg", 0.4, True)
# #img104_4_166_3 = im_stitcher("../input/test_sm/set104_4.jpeg", "../input/test_sm/set166_3.jpeg", 0.4, True)
# #img104_5_166_2 = im_stitcher("../input/test_sm/set104_5.jpeg", "../input/test_sm/set166_2.jpeg", 0.4, True)

# #plt.imsave('Set104_1_166_1_BRISK_matching.jpeg',img104_1_166_1) 
# #plt.imsave('Set104_2_166_4_BRISK_matching.jpeg',img104_2_166_4) 
# #plt.imsave('Set104_3_166_5_BRISK_matching.jpeg',img104_3_166_5) 
# #plt.imsave('Set104_4_166_3_BRISK_matching.jpeg',img104_4_166_3) 
# #plt.imsave('Set104_6_166_2_BRISK_matching.jpeg',img104_5_166_2) 

# #img1_1_85_5 = im_stitcher("../input/test_sm/set1_1.jpeg", "../input/test_sm/set85_5.jpeg", 0.4, True)
# #img1_2_85_4 = im_stitcher("../input/test_sm/set1_2.jpeg", "../input/test_sm/set85_4.jpeg", 0.4, True)
# #img1_3_85_2 = im_stitcher("../input/test_sm/set1_3.jpeg", "../input/test_sm/set85_2.jpeg", 0.4, True)
# #img1_4_85_3 = im_stitcher("../input/test_sm/set1_4.jpeg", "../input/test_sm/set85_3.jpeg", 0.4, True)
# #img1_5_85_1 = im_stitcher("../input/test_sm/set1_5.jpeg", "../input/test_sm/set85_1.jpeg", 0.4, True)

# #plt.imsave('Set1_1_85_5_BRISK_matching.jpeg',img1_1_85_5) 
# #plt.imsave('Set1_2_85_4_BRISK_matching.jpeg',img1_2_85_4) 
# #plt.imsave('Set1_3_85_2_BRISK_matching.jpeg',img1_3_85_2) 
# #plt.imsave('Set1_4_85_3_BRISK_matching.jpeg',img1_4_85_3) 
# #plt.imsave('Set1_5_85_1_BRISK_matching.jpeg',img1_5_85_1) 

# #img3_1_22_1 = im_stitcher("../input/test_sm/set3_1.jpeg", "../input/test_sm/set22_1.jpeg", 0.4, True)
# #img3_2_22_2 = im_stitcher("../input/test_sm/set3_2.jpeg", "../input/test_sm/set22_2.jpeg", 0.4, True)
# #img3_3_22_5 = im_stitcher("../input/test_sm/set3_3.jpeg", "../input/test_sm/set22_5.jpeg", 0.4, True)
# #img3_4_22_3 = im_stitcher("../input/test_sm/set3_4.jpeg", "../input/test_sm/set22_3.jpeg", 0.4, True)
# #img3_5_22_4 = im_stitcher("../input/test_sm/set3_5.jpeg", "../input/test_sm/set22_4.jpeg", 0.4, True)

# #plt.imsave('Set3_1_22_1_BRISK_matching.jpeg',img3_1_22_1) 
# #plt.imsave('Set3_2_22_2_BRISK_matching.jpeg',img3_2_22_2) 
# #plt.imsave('Set3_3_22_5_BRISK_matching.jpeg',img3_3_22_5) 
# #plt.imsave('Set3_4_22_3_BRISK_matching.jpeg',img3_4_22_3) 
# #plt.imsave('Set3_5_22_4_BRISK_matching.jpeg',img3_5_22_4) 

# #img5_1_68_3 = im_stitcher("../input/train_sm/set5_1.jpeg", "../input/test_sm/set68_3.jpeg", 0.4, True)
# #plt.imsave('Set5_1_68_3_BRISK_matching.jpeg',img5_1_68_3) 

# img160_5_74_1 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_1.jpeg", 1, True)
# img160_5_74_2 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_2.jpeg", 0.4, True)
# img160_5_74_3 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_3.jpeg", 0.4, True)
# img160_5_74_4 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_4.jpeg", 0.4, True)
# img160_5_74_5 = im_stitcher("../input/train_sm/set160_5.jpeg", "../input/test_sm/set74_5.jpeg", 0.4, True)

# plt.imsave('Set160_5_74_1_BRISK_matching.jpeg',img160_5_74_1) 
# plt.imsave('Set160_5_74_2_BRISK_matching.jpeg',img160_5_74_2) 
# plt.imsave('Set160_5_74_3_BRISK_matching.jpeg',img160_5_74_3) 
# plt.imsave('Set160_5_74_4_BRISK_matching.jpeg',img160_5_74_4) 
# plt.imsave('Set160_5_74_5_BRISK_matching.jpeg',img160_5_74_5) 

plt.imsave("160_5",cv2.imread("../input/train_sm/set160_5.jpeg"))
plt.imsave("74_1",cv2.imread("../input/test_sm/set74_1.jpeg"))



