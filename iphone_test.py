import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats
import sys
import time
import math
import cPickle as pickle

def get_angle(x,y):
    r = math.sqrt(x**2+y**2)
    if x>0:
        theta = math.atan(y/x)
    else:
        theta = math.atan(y/x)+math.pi
    theta = theta % (2*math.pi)
    theta = (theta+scipy.pi)%(2*scipy.pi)
    return theta


def order_by_area(contour_list):
    """
    Given a list of contours finds the contour with the maximum area and
    returns
    """
    contour_areas = np.array([cv2.contourArea(c) for c in contour_list])
    indices = list(reversed(np.argsort(contour_areas)))
    reordered_list = [contour_list[index] for index in indices]
    return reordered_list

def get_contour_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX,cY

def distance(x_1,y_1,x_2,y_2):
    return math.sqrt((x_2-x_1)**2+(y_2-y_1)**2)

def get_contour_center_distance(contour1,contour2):
    cX_1,cY_1 = get_contour_center(contour1)
    cX_2,cY_2 = get_contour_center(contour2)
    return distance(cX_1,cY_1,cX_2,cY_2)

video = sys.argv[2]
vidcap = cv2.VideoCapture(video)
success,image = vidcap.read()
image_height,image_width,_ = np.shape(image)

pos_angle_array = np.full((10000,3),np.nan)

plt.ion()
plt.figure(1)
im = plt.imshow(image)#,cmap='gray')

center_xs,center_ys = [0,0],[0,0]
centers = plt.scatter(center_xs,center_ys,color='red')
angless = plt.figtext(0.5,0.5,' ',color='r',horizontalalignment='center')


row = 0
while success:
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plt.draw()
    plt.pause(0.01)
    # threshold_image = cv2.adaptiveThreshold(image1, np.iinfo(image.dtype).max,\
         # cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,251,2)
    rval, threshold_image = cv2.threshold(image1,220,255,cv2.THRESH_BINARY)
    contour_list, _ = cv2.findContours(threshold_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour_list = order_by_area(contour_list)
    if len(contour_list)<1:
        print('no dots detected')
    else:
        #Toss aside false contours
        try:
            while get_contour_center_distance(contour_list[0],contour_list[1])>0.1*image_width:
                contour_list = contour_list[1:]

        except(ZeroDivisionError,IndexError):
            print('no dots detected')
            pass

        im.set_data(image)

        try:
            center_x_1,center_y_1 = get_contour_center(contour_list[0])
            center_x_2,center_y_2 = get_contour_center(contour_list[1])
            center_coords = np.array([[center_x_1,center_y_1],[center_x_2,center_y_2]])
            cv2.line(image,(center_x_1,center_y_1),(center_x_2,center_y_2),(255,0,0), 2)

            midpoint = (center_x_1+center_x_2)/2,(center_y_1+center_y_2)/2
            angle = get_angle(float(
            center_x_2-center_x_1),float(center_y_2-center_y_1))
            if np.isnan(angle):
                print('alert')
                time.sleep(2)
            pos_angle_array[row,:]= midpoint[0],midpoint[1],angle
            centers.set_offsets(center_coords)
            angless.set_text(str(angle))
            cv2.drawContours(image,contour_list[0:2],-1,(0,0,255),2)

        except(ZeroDivisionError,IndexError):
            pass

    #after everything is done load next frame
    success,image = vidcap.read()
    row+=1

output_file = sys.argv[1]+'.pkl'
with open(output_file, 'w') as f:
    pickle.dump(pos_angle_array,f)
