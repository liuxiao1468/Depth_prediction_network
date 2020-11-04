
import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
from numpy import inf
import time

NAME = "training-RGB-{}".format(int(time.time()))
NAME0 = "training-depth-{}".format(int(time.time()))
frame_width = 640
frame_height = 360
out0 = cv2.VideoWriter(NAME+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
    # init_params.coordinate_units = sl.UNIT.INCH  # use inches
    # init_params.camera_fps = 2  # fps
    init_params.camera_fps = 2  # fps
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.4
    init_params.depth_maximum_distance = 20
    seconds = 600
    time = seconds * init_params.camera_fps


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # Use STANDARD sensing mode

    # Capture 50 images and depth, then stop
    i = 0
    flag = 0
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    test_depth = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

    while i < time:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)

            image_ocv = image.get_data()
            scale_percent = 50

            #calculate the 50 percent of original dimensions
            width = int(image_ocv.shape[1] * scale_percent / 100)
            height = int(image_ocv.shape[0] * scale_percent / 100)

            # dsize
            dsize = (width, height)

            # resize image
            image_ocv = cv2.resize(image_ocv, dsize)

            # cv2.imshow('depth', real_depth)
            cv2.imshow('Image', image_ocv)
            key = cv2.waitKey(1)
            if key == ord('r'):
                # "r" pressed
                print("recording")
                flag = 1
            if key == ord('s'):
                # "r" pressed
                flag = 0
                print("stoped")
            if flag == 1:
            	out0.write(image_ocv)

            # Increment the loop
            i = i + 1
            sys.stdout.flush()
    out.release()
    cv2.destroyAllWindows()
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
