import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the dimensions of checkerboard 
CHECKERBOARD = (6, 8) 

# Vector for 3D points 
threedpoints = [] 
  
# Vector for 2D points 
twodpoints = [] 

#  3D points real world coordinates 
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1],  3), np.float32)

objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


# Extracting path of individual image stored 
# in a given directory. Since no path is 
# specified, it will take current directory 
# jpg files alone 
images = glob.glob('./images/*.png')

for filename in images: 
    image = cv.imread(filename) 
    grayColor = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
  
    # Find the chess board corners 
    # If desired number of corners are 
    # found in the image then ret = true 
    ret, corners = cv.findChessboardCorners( 
                    grayColor, CHECKERBOARD,  
                    cv.CALIB_CB_ADAPTIVE_THRESH  
                    + cv.CALIB_CB_FAST_CHECK + 
                    cv.CALIB_CB_NORMALIZE_IMAGE) 
  
    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret == True: 
        threedpoints.append(objectp3d) 
  
        # Refining pixel coordinates 
        # for given 2d points. 
        corners2 = cv.cornerSubPix( 
            grayColor, corners, (11, 11), (-1, -1), criteria) 
  
        twodpoints.append(corners2) 
  
        # Draw and display the corners 
        image = cv.drawChessboardCorners(image,  
                                          CHECKERBOARD,  
                                          corners2, ret) 
  
    cv.imshow('img', image) 
    cv.waitKey(0) 
  
cv.destroyAllWindows() 
  
h, w = image.shape[:2]

# Perform camera calibration by 
# passing the value of above found out 3D points (threedpoints) 
# and its corresponding pixel coordinates of the 
# detected corners (twodpoints) 
ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera( 
    threedpoints, twodpoints, grayColor.shape[::-1], None, None) 

# Displaying required output 
print(" Camera matrix:") 
print(matrix) 
  
print("\n Distortion coefficient:") 
print(distortion) 
  
print("\n Rotation Vectors:") 
print(r_vecs) 
  
print("\n Translation Vectors:") 
print(t_vecs)


############## UNDISTORTION #####################################################

img = cv.imread('./images/img0.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, matrix, distortion, None, newCameraMatrix)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('./images/calibresult.png', dst)

# Reprojection Error
mean_error = 0
for i in range(len(threedpoints)):
    imgpoints2, _ = cv.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv.norm(twodpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "\ntotal error: {}".format(mean_error/len(threedpoints)) )

