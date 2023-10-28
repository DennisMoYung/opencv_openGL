import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

#read img
img1 = cv.imread("right.jpg")
img1G = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2 = cv.imread("left.jpg")
img2G = cv.cvtColor(img2 ,cv.COLOR_BGR2GRAY)

#start sift process
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1G,None)
kp2, des2 = sift.detectAndCompute(img2G,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

    
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape[:-1]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    #transform shape
    dst = cv.perspectiveTransform(pts,M)
    img2G = cv.polylines(img2G,[np.int32(dst)],True,255,3, cv.LINE_AA)

    #draw matched feature point 
    img1G = cv.drawKeypoints(img1G, kp1,img1G,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2G = cv.drawKeypoints(img2G, kp2,img2G,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('sift_keypoints_right.jpg',img1G)
    cv.imwrite('sift_keypoints_left.jpg',img2G)
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask, # draw only inliers
        flags = 2)
    img_ransac = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv.imwrite('ransac.jpg', img_ransac)

else:
  raise AssertionError("Can’t find enough keypoints.")

#output image
dst = cv.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
plt.subplot(122),plt.imshow(dst),plt.title("Warped Image")
plt.show()
plt.figure()
dst[0:img2.shape[0], 0:img2.shape[1]] = img2
cv.imwrite("output.jpg",dst)
plt.imshow(dst), plt.title("output")
plt.show()