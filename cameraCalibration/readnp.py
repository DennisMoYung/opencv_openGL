import numpy as np

cameraMatrix_file_path = r'D:\document\opencv_openGL\cameraCalibration\cameraMatrix.npy'
distCoeffs_file_path = r'D:\document\opencv_openGL\cameraCalibration\distCoeffs.npy'

# Load the data using the file paths
cameraMatrix = np.load(cameraMatrix_file_path)
distCoeffs = np.load(distCoeffs_file_path)

# Print the data
print("Camera Matrix:\n", cameraMatrix)
print("\nDistortion Coefficients:\n", distCoeffs)


#Camera Matrix:
#[[fx,  0, cx],
# [ 0, fy, cy],
# [ 0,  0,  1]]
#
#Distortion Coefficients:
# [k1, k2, p1, p2, k3]
# k1, k2, k3: Radial distortion coefficients
# p1, p2: Tangential distortion coefficients.