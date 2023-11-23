import cv2
import numpy as np
import glob

# Define the checkerboard size
CHECKERBOARD = (6, 8)  # Number of inner corners per a checkerboard row and column

# Arrays to store object points and image points.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ..., (5,7,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Loop over all calibration images
images = glob.glob('path/to/calibration/images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        
#This step computes the camera matrix, distortion coefficients, rotation and translation vectors etc.
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#calculating the reprojection error, which measures how close the projected points are to the actual image points.
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("Total error: ", mean_error / len(objpoints))


# Save the camera matrix and distortion coefficients
np.save("cameraMatrix.npy", cameraMatrix)
np.save("distCoeffs.npy", distCoeffs)
