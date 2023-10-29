import cv2
import numpy as np

def compute_homography(frame1, frame2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(frame1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(frame2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homography

# Capture reference frames
cap1 = cv2.VideoCapture('path_to_video1.mp4')
cap2 = cv2.VideoCapture('path_to_video2.mp4')

ret1, ref_frame1 = cap1.read()
ret2, ref_frame2 = cap2.read()

# Compute and store the homography
H = compute_homography(ref_frame1, ref_frame2)

def stitch_frames_using_precomputed_homography(frame1, frame2, H):
    result = cv2.warpPerspective(frame1, H, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))
    result[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
    return result

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('stitched_video.avi', fourcc, 20.0, (output_width, output_height))

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    stitched_frame = stitch_frames_using_precomputed_homography(frame1, frame2, H)
    out.write(stitched_frame)

cap1.release()
cap2.release()
out.release()
