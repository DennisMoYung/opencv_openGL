import numpy as np
import cv2 as cv
import glob

from scipy.spatial.transform import Rotation 

##for PnP
def draw(img, corners, imgpts):

    corner = tuple(corners[0].astype(int).ravel())
    img = cv.line(img, corner, tuple(imgpts[0].astype(int).ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].astype(int).ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].astype(int).ravel()), (0,0,255), 5)

    return img

def drawBoxes(img, corners, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img



##cameraCalibration

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
images = glob.glob('./images/*.jpeg')

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
        cv.imwrite(f"{filename[:-4]}_corner.png", image)
  
    cv.imshow('img', cv.pyrDown(image, dstsize=(image.shape[1] // 2, image.shape[0] // 2))) 
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
  
# print("\n Rotation Vectors:") 
# print(r_vecs) 
  
# print("\n Translation Vectors:") 
# print(t_vecs)

np.savetxt('data/camera.txt', matrix)
np.savetxt('data/distortion.txt', distortion)

# Reprojection Error
mean_error = 0
for i in range(len(threedpoints)):
    imgpoints2, _ = cv.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv.norm(twodpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "\ntotal error: {}".format(mean_error/len(threedpoints)) )
print()


##PnP
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])


for image in glob.glob('./images/*.jpeg'):

    img = cv.imread(image)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (8, 6),None)

    if ret == True:

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, matrix, distortion)

        print("Rotation matrix:")
        #print(rvecs, "\n" )

        rotation_matrix = np.zeros(shape=(3,3))
        cv.Rodrigues(rvecs, rotation_matrix)
        print(rotation_matrix)
        # print()
        r =  Rotation.from_matrix(rotation_matrix)
        print("Rotation angle: (degree)")
        angles = r.as_euler("zyx", degrees=True)


        print(angles)
        print()
        print("Tvecs")
        print(tvecs)

        # Project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, matrix, distortion)

        img = draw(img,corners2,imgpts)
        cv.imshow('img',cv.pyrDown(img, dstsize=(img.shape[1] // 2, img.shape[0] // 2)))

        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite('pose'+image, img)

        

        



cv.destroyAllWindows()

