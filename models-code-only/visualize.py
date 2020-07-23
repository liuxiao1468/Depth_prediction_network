import numpy as np
import cv2

# training-depth-1594692293.avi  training-RGB-1594692293.avi
# training-depth-1594692373.avi
# training-RGB-1594692523.avi training-depth-1594692523.avi
# 'training-RGB-1594692013'
# training-RGB-1594692626

name_1 = 'training-RGB-1594692626'
name_2 = 'training-depth-1594692626'

vidcap = cv2.VideoCapture(name_1+'.avi')
vidcap1 = cv2.VideoCapture(name_2+'.avi')
success,image = vidcap.read()
success1,image1 = vidcap1.read()
count = 0
path1 = 'train/'+name_1
path2 = 'ground_truth/'+name_2

while success and success1:
    if count >= 1:
        # print(image.shape)
        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2))) 
        image1 = cv2.resize(image1, (int(image1.shape[1]/2), int(image1.shape[0]/2)))
        # image1 = (image1/255)*6
        cv2.imwrite(path1+'-%d.jpg' % count, image)     # save frame as JPEG file      
        cv2.imwrite(path2+'-%d.jpg' % count, image1)     # save frame as JPEG file  
        success,image = vidcap.read()
        success1,image1 = vidcap1.read()
        print('Read a new frame: ', success)
    count += 1
    # if count == 10:
    #   break



# # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# img0 = cv2.imread('frame-4.jpg')
# img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# print(img.shape)
# depth = img[50][50].astype(float)
# print(depth)
# distance = (depth/255)*6
# print(distance)




# distance_img =(img/255)*6
# print(distance_img)

# arr = distance_img
# arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')


# # Window name in which image is displayed 
# window_name = 'ground_truth'
  
# cv2.imshow(window_name, distance_img)
# cv2.imshow('raw', arr)
# cv2.imwrite('frame-3_truth.jpg', arr)     # save frame as JPEG file  
  
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#   