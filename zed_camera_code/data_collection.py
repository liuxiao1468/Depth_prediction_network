########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
from numpy import inf
import time


NAME = "training-RGB-{}".format(int(time.time()))
NAME0 = "training-depth-{}".format(int(time.time()))

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
    # init_params.coordinate_units = sl.UNIT.INCH  # use inches
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
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            zed.retrieve_image(test_depth, sl.VIEW.DEPTH)

            
            ocv_depth = test_depth.get_data()
            image_ocv = image.get_data()



            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            x = round(image.get_width() / 2)
            y = round(image.get_height() / 2)
            x1 = round(x/2)
            y1 = round(y/2)
            x2 = x+x1
            y2 = y+y1
            # print(image.get_width())
            # print(image.get_height())

            err, point_cloud_value = point_cloud.get_value(x, y)
            depth_value = depth.get_value(y,x)
            real_depth = depth.get_data()
            real_depth = np.where(real_depth==inf, 20.0, real_depth)
            real_depth = np.where(real_depth== -inf, 0.4, real_depth)

            # print(real_depth.shape)
            depth_value_1 = depth.get_value(y1,x1)
            depth_value_2 = depth.get_value(y2,x2)

            cv2.circle(ocv_depth, (int(x), int(y) ), 1, (0,0,255), thickness=3)
            cv2.circle(ocv_depth, (int(x1), int(y1) ), 1, (0,0,255), thickness=3) 
            cv2.circle(ocv_depth, (int(x2), int(y2) ), 1, (0,0,255), thickness=3)

            print('depth_value: ',depth_value_1[1]," ", depth_value[1], " ", depth_value_2[1])
            print('real_depth: ', real_depth[y,x], "max distance: ", np.amax(real_depth) )

            # cv2.imshow('depth', real_depth)
            cv2.imshow('Image', ocv_depth)
            key = cv2.waitKey(10)
            if key == ord('r'):
                # "r" pressed
                flag = 1
            if key == ord('s'):
                # "r" pressed
                flag = 0
            if flag == 1:
                real_depth = real_depth*(255/20)
                depth_image = np.uint8(real_depth)
                cv2.imwrite('/home/xiaoliu/zed_camera/test_data/RGB/'+NAME+str(i)+'-.png',image_ocv)
                cv2.imwrite('/home/xiaoliu/zed_camera/test_data/Depth/'+NAME0+str(i)+'-.png',depth_image)
                print('RECORDING')

            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])

            point_cloud_np = point_cloud.get_data()
            point_cloud_np.dot(tr_np)

            if not np.isnan(distance) and not np.isinf(distance):
                distance = round(distance)
                print("Distance to Camera at ({0}, {1}): {2} m\n".format(x, y, distance))
                # Increment the loop
                i = i + 1
            else:
                print("Can't estimate distance at this position, move the camera\n")
            sys.stdout.flush()

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
