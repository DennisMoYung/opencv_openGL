import numpy as np

file = r'D:\document\opencv_openGL\cameraCalibration\CalibrationMatrix_cpt.npz'

# Load the data using the file paths

data = np.load(file)

camera_matrix = data['Camera_matrix']
dist_coeffs = data['distCoeff']
rotational_v = data['RotationalV']
translation_v = data['TranslationV']

# Print the data
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)
print("\nRotational Vector:\n", rotational_v)
print("\nTranslation Vector:\n", translation_v)

data.close()
#Camera Matrix:
#[[fx,  0, cx],
# [ 0, fy, cy],
# [ 0,  0,  1]]

#Distortion Coefficients:
# [k1, k2, p1, p2, k3]
# k1, k2, k3: Radial distortion coefficients
# p1, p2: Tangential distortion coefficients.

# Rotational Vector:
# [r1, r2, r3]

# Translation Vector:
# [t1, t2, t3]