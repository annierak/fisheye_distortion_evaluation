import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats
import sys
import time
import math
import cPickle as pickle


def distance(x_1,y_1,x_2,y_2):
    return math.sqrt((x_2-x_1)**2+(y_2-y_1)**2)

def next_grid_point(x_grid_size,y_grid_size,current_x,current_y):
    #advance to the next center
    if current_x<x_grid_size:
        return (current_x+1,current_y)
    else:
        return (1,current_y+1)

detected = sys.argv[1]
input_file_hat = detected+'.pkl'

original_data_file = sys.argv[2]
original_data = np.loadtxt(original_data_file,delimiter=',')
x_originals,y_originals,angle_originals = original_data[:,0],\
    original_data[:,1],original_data[:,2]


with open(input_file_hat,'r') as File:
    detected_values = pickle.load(File)

# x_grid_size = 15
x_grid_size = len(np.unique(x_originals))
y_grid_size = len(np.unique(y_originals))
#known_angles = 50
print(np.unique(angle_originals))
raw_input('done?')
known_angles = len(np.unique(angle_originals))

new_center_threshold = 30 #number of pixels distance to count as a new grid point

sorted_detected_values = np.full((y_grid_size,x_grid_size,known_angles,200),np.nan)
#this is y coord by x coord by known angle by collected angle_hat

angle_switch_frames = 4 #Threshold for determining the angle is switching
x_hats,y_hats,angle_hats = detected_values[:,0],detected_values[:,1],detected_values[:,2]

#cutoff empty values at the beginning
counter = 0
while np.isnan(angle_hats[counter]):
    counter +=1
x_hats,y_hats,angle_hats = x_hats[counter:],y_hats[counter:],angle_hats[counter:]

nans_inarow = 1
x_grid_point = 1
y_grid_point = 1
known_angle_index = 0
previous_x_center = x_hats[0]
previous_y_center = y_hats[0]
observed_angle_counter = 0
last_jump_index = 0 #for indexing angle_hats

for i in range(len(angle_hats)):
    # print i
    # print(angle_hats[i:i+50])
    # time.sleep(1)
    #check if there is a center jump
    if distance(x_hats[i],y_hats[i],previous_x_center,previous_y_center
    )>new_center_threshold:
        #and move to the next grid point if so
        x_grid_point,y_grid_point = next_grid_point(
            x_grid_size,y_grid_size,x_grid_point,y_grid_point)
        known_angle_index = 0
        observed_angle_counter = 0
        #the current centers become the previous centers
        previous_x_center = x_hats[i]
        previous_y_center = y_hats[i]

        #if we're advancing to a new grid point, make a histogram of the
        #observed angles to check that we don't have gaps
        observed_angles = angle_hats[last_jump_index:i]
        observed_angles = observed_angles[~np.isnan(observed_angles)]
        plt.hist(observed_angles,bins=200)
        plt.show()
        last_jump_index = i


    #check for a angle advancement using the blackout phase
    if np.isnan(angle_hats[i]):
        nans_inarow+=1
    else:
        nans_inarow = 1
    #if there is an angle advancement jump to the next known angle
    if ((nans_inarow==angle_switch_frames) or (
    #if there is a particularly long gap jump again
    nans_inarow==3*angle_switch_frames)):
        known_angle_index +=1
        # print('------advancing angle------')
        # print(angle_hats[i-3:i+3])
        # time.sleep(0.1)
        print(known_angle_index)
        time.sleep(0.1)
        if known_angle_index>known_angles-1:
            known_angle_index = 0
        observed_angle_counter = 0




    #now that we know we're in the right x_grid_point, y_grid_point
    #and known_angle_index, populate this row with the observed angles
    #only if the entry is not nan
    if np.isnan(angle_hats[i]):
        continue
    else:
        sorted_detected_values[y_grid_point-1,x_grid_point-1,
            known_angle_index,observed_angle_counter] = angle_hats[i]
        observed_angle_counter+=1
        # print(y_grid_point-1,x_grid_point-1,
        #     known_angle_index,observed_angle_counter)

# for known_angle_index in range(known_angles):
#     print(sorted_detected_values[0,0,known_angle_index,:])
    # time.sleep(0.1)

plt.figure()
for i in range(2):#x_grid_size):
    for j in range(1):#y_grid_size):
        # plt.subplot2grid((y_grid_size,x_grid_size),(j,i))
        plt.figure()
        for known_angle_index in range(known_angles):
            observed_angles = sorted_detected_values[j,i,known_angle_index,:]
            observed_angles = observed_angles[~np.isnan(observed_angles)]
            # time.sleep(1)
            plt.plot([0,2*math.pi],[0,2*math.pi],'r')
            plt.scatter(np.full(len(observed_angles),
            known_angle_index*2*math.pi/(known_angles+1)),
            observed_angles,c='blue')
                # print(known_angle_index*2*math.pi/(known_angles+1))
                # print(observed_angles)
plt.show()
