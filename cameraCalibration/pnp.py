import numpy as np
import cv2 as cv
import glob
from scipy.spatial.transform import Rotation 


file = r'D:\document\opencv_openGL\cameraCalibration\CalibrationMatrix_cpt.npz'

# Load the data using the file paths

data = np.load(file)

camera_matrix = data['Camera_matrix']
dist_coeffs = data['distCoeff']
rotational_v = data['RotationalV']
translation_v = data['TranslationV']

mtx = camera_matrix.reshape(-1,3)
dist = dist_coeffs



def resize_image(image, max_size=800):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scaling_factor = max_size / float(h if h > w else w)
        image = cv.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
    return image


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


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])


for image in glob.glob(r'D:\document\opencv_openGL\cameraCalibration\images\*.jpg'):

    img = cv.imread(image)
    img = resize_image(img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (8, 6),None)

    if ret == True:

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)

        print("R :")
        print(rvecs, "\n" )

        rotation_matrix = np.zeros(shape=(3,3))
        cv.Rodrigues(rvecs, rotation_matrix)
        print(rotation_matrix)
        print()
        r =  Rotation.from_matrix(rotation_matrix)
        angles = r.as_euler("zyx",degrees=True)
        #### Modify the angles
        print(angles)
        print()

        # Project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv.imshow('img',img)

        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite('pose'+image, img)

        

        



cv.destroyAllWindows()