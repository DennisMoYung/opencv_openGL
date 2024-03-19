from threading import Thread
import cv2, time, numpy as np

#real-time capturing and 

#please enter the setting by your own

ip_l = ""
port_l = ""
username_l = ""
pwd_l = ""

ip_r = ""
port_r = ""
username_r = ""
pwd_r = ""

#default setting for own usage, you can change it by yourself
last = np.array([[  5.88427416e-01 , -2.30387958e-02 , 4.85086939e+02], [ 1.93747521e-02 , 9.46848431e-01 , 2.73486166e+00], [ -3.98097006e-04 ,-4.64597427e-05 , 1.00000000e+00]])

last_x = 1171
last_y = 489


url_l = f"rtsp://{username_l}:{pwd_l}@{ip_l}:{port_l}/stream2"
url_r = f"rtsp://{username_r}:{pwd_r}@{ip_r}:{port_r}/stream2"
 
class VideoStream(object):
    def __init__(self, src=url_l):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            else:
                exit()
            time.sleep(.01)
    
    def get_frame(self):
        return self.frame
    
    def get_isOpen(self):
        return self.capture.isOpened()


if __name__ == '__main__':
    video_stream_l = VideoStream()
    video_stream_r = VideoStream(url_r)

    time.sleep(0.3)
    img_l = video_stream_l.get_frame()
    img_r = video_stream_r.get_frame()

    img_l_g = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    img_r_g = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_l_g,None)
    kp2, des2 = sift.detectAndCompute(img_r_g,None)

    good = []
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    for m, n in matches:
        if m.distance < (0.7 * n.distance):
            good.append((kp2[m.trainIdx].pt, kp1[m.queryIdx].pt))

    src_pts = []
    dst_pts = []
    for src, dst in good :
        src_pts.append(src)
        dst_pts.append(dst)
    
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    best_H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,ransacReprojThreshold=3, maxIters=1000, confidence=0.85)

    h1, w1 = img_l.shape[:2]
    h2, w2 = img_r.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, best_H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]

    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    Mt = Ht @ best_H
    x_diff = xmax - xmin
    y_diff = ymax - ymin


    while True:

        isOpen_l = video_stream_l.get_isOpen()
        isOpen_r = video_stream_r.get_isOpen()
        if(not (isOpen_l and isOpen_r)):
            exit()

        try:
            img_l = video_stream_l.get_frame()
            img_r = video_stream_r.get_frame()

            result = cv2.warpPerspective(img_r, Mt, (x_diff, y_diff), flags=cv2.INTER_NEAREST)
            result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img_l
            cv2.imshow('result',result)

            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(1)

        except AttributeError:
            pass
