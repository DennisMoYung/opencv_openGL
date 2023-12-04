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

# Adjust the path to the folder where your images are stored
images = glob.glob(r'D:\document\opencv_openGL\cameraCalibration\images\*.jpg')

# Check if images are found
if not images:
    raise ValueError("No images found in the specified path.")

last_img_shape = None

# Loop over all calibration images
for fname in images:
    print(f"Processing image: {fname}")  # Debugging message
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points
    if ret:
        print("Checkerboard detected!")  # Debugging message
        objpoints.append(objp)
        imgpoints.append(corners)
        last_img_shape = gray.shape[::-1]
    else:
        print("Checkerboard not detected.")  # Debugging message

# Check if any checkerboard was detected
if not objpoints:
    raise ValueError("Checkerboard not detected in any image.")

# Camera calibration
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, last_img_shape, None, None)

# Reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total error: ", mean_error / len(objpoints))

# Save the camera matrix and distortion coefficients
np.save(r"D:\document\opencv_openGL\cameraCalibration\cameraMatrix.npy", cameraMatrix)
np.save(r"D:\document\opencv_openGL\cameraCalibration\distCoeffs.npy", distCoeffs)

np.savez(
    f"D:\document\opencv_openGL\cameraCalibration\CalibrationMatrix_cpt",
    Camera_matrix=cameraMatrix,
    distCoeff=distCoeffs,
    RotationalV=rvecs,
    TranslationV=tvecs,
)