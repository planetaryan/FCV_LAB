import cv2

cap = cv2.VideoCapture("/home/student/220962110/220962110_CV_28/Lab1/video.mp4")

while(cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:

        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break


cap.release()
cv2.destroyAllWindows()






