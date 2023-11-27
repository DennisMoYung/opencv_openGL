import numpy as np
import cv2 as cv
import glob
from scipy.spatial.transform import Rotation 



mtx = np.array([[530.43495126, 0, 326.8957335] ,[0, 534.36359692, 247.02114604], [0, 0, 1]]).reshape(-1,3)
dist = np.array([[ -0.03195748, 0.2449868,  0.00334185,  0.00221634 ,-0.51086226]])

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


for image in glob.glob('./images/*.png'):

    img = cv.imread(image)
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