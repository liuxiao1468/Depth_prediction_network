import pyrealsense2 as rs
import cv2
import numpy as np
import time

# Create a pipeline
pipeline = rs.pipeline()
frame_width = 640
frame_height = 360

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 3  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


NAME = "training-RGB-{}".format(int(time.time()))
NAME0 = "training-depth-{}".format(int(time.time()))
out0 = cv2.VideoWriter(NAME+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
out1 = cv2.VideoWriter(NAME0+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height),0)


# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame().as_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_frame1 = frames.get_depth_frame()
        color_frame1 = frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame or not depth_frame1 or not color_frame1:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image1 = np.asanyarray(depth_frame1.get_data())
        color_image1 = np.asanyarray(color_frame1.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)


        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        depth_image_3d1 = np.dstack((depth_image1, depth_image1, depth_image1))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image1, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, bg_removed))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        cv2.imshow('Example RGB', color_image)
        cv2.imshow('Example depth', depth_colormap)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if  key & 0xFF == ord('q'):
            
            break
        if key == ord('r'):
            # SPACE pressed
            depth_image = depth_image*depth_scale
            print(np.max(depth_image))
            print(np.min(depth_image))
            # depth_image = ((depth_image - np.min(depth_image))/(np.max(depth_image)-np.min(depth_image)))*255
            depth_image = depth_image*(255/6)
            depth_image = np.uint8(depth_image)
            out0.write(color_image)
            out1.write(depth_image)
            print('RECORDING')
    out.release()
    cv2.destroyAllWindows()
finally:
    pipeline.stop()