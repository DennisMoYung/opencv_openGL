import cv2
import numpy as np

# Step 1: Read input images
image1 = cv2.imread("path_to_image1.jpg")
image2 = cv2.imread("path_to_image2.jpg")

# Step 2 & 3: Detect keypoints and match them
# We'll use SIFT for this example
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Use FLANN based matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Filter good matches
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# Step 4: Calculate the homography
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Step 5: Warp and stitch the images
result = cv2.warpPerspective(image1, M, (image1.shape[1] + image2.shape[1], image1.shape[0]))
result[0:image2.shape[0], 0:image2.shape[1]] = image2

# Save the result
cv2.imwrite('stitched_image.jpg', result)
