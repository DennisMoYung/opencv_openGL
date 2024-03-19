import cv2 as cv

#demo for display video capture (non-synchronize)

##please enter the data by your own
ip_l = ""
port_l = ""
username_l = ""
pwd_l = ""

ip_r = ""
port_r = ""
username_r = ""
pwd_r = ""



url_l = f"rtsp://{username_l}:{pwd_l}@{ip_l}:{port_l}/stream2"
url_r = f"rtsp://{username_r}:{pwd_r}@{ip_r}:{port_r}/stream2"

cap_l = cv.VideoCapture(url_l)
cap_r = cv.VideoCapture(url_r)

while True:
    ret_l, img_l = cap_l.read()
    ret_r, img_r = cap_r.read()

    if ret_l == True and ret_r == True:
        cv.imshow('realtime image', cv.hconcat((img_l, img_r)))
        k = cv.waitKey(10)& 0xff
        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv.imwrite('right_camera.png', img_r)
            cv.imwrite('left_camera.png', img_l)
            print("image saved!")
    else:
        break

cap_l.release()
cap_r.release()
cv.destroyAllWindows()