import matplotlib.pyplot as plt
import cv2
import numpy as np
import time


width = int(1080/2)
height = int(720/2)

name_1 = 'training-RGB-1604607562'

vidcap = cv2.VideoCapture(name_1+'.avi')

success,image = vidcap.read()
print(image.shape)
height,width,_ = image.shape


NAME = "walking_1_result-{}".format(int(time.time()))
out0 = cv2.VideoWriter(NAME+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))



while (vidcap.isOpened() ):
    success,image = vidcap.read()
    if success:
        # RGB-image operation
        # image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
        # Depth-image operation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        out0.write(image)
        cv2.namedWindow('Align video', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align video', image)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
out0.release()
vidcap.release()

cv2.destroyAllWindows()