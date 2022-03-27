'''
RPi Pinouts

I2C Pins 

GPIO2 -> SDA
GPIO3 -> SCL

'''

from curses import baudrate
from matplotlib.widgets import Widget
import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd
import serial
import time

# Serial Variables
serial_device = "/dev/ttyACM0"
baudrate = 9600
timeout=1

# Object Detection
# Reference: https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1
detector = hub.load("efficientdet_lite2_detection_1")
labels = pd.read_csv('labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']
video_device = 0
cap = cv2.VideoCapture(video_device)
w = 500
h = w
cx = w/2
cy = h/2
    

#############################################################################################
#                                    Movement Logic
#############################################################################################


def priority_coordinates(x_dots,y_dots):
    o_distances = []
    for i in range(0,len(x_dots)):
        distance = ((w - x_dots[i])**2 + (h - y_dots[i])**2)**0.5
        o_distances.append(round(distance,3))
    p_distances = sorted(o_distances)
    p_coordinates = o_distances.index(p_distances[0])
    return x_dots[p_coordinates],y_dots[p_coordinates]

# Set the PID values for x and y axis movement. we set I,D values as 0 for proportional controller.
Px,Ix,Dx=-1/160,0,0
Py,Iy,Dy=-0.2/120,0,0

def coordinates_2_power(x,y):
    # If a red dot is near the centre of the frame, move the drone forward
    if x <= cx + 15 and x >= cx - 15 and y <= cy + 15 and y >= cy - 15:
        # Z axis movement
        pf_b = 50
        # X axis movement
        pr_l = 0
        # Y axis movement
        pu_d = 0
        return pf_b,pr_l,pu_d
    else:

        #calculate pixels to move 
        error_x=160-x  # X-coordinate of Centre of frame is 160
        error_y=120-y # Y-coordinate of Centre of frame is 120

        # Calculate the deviation of the face from the centre and set it as error
    
        integral_x=integral_x+error_x
        integral_y=integral_y+error_y
        differential_x= prev_x- error_x
        differential_y= prev_y- error_y
        prev_x=error_x
        prev_y=error_y

        # Calculate the value for x movement and y movement using PID logic.(valx,valy)

        valx=Px*error_x +Dx*differential_x + Ix*integral_x
        valy=Py*error_y +Dy*differential_y + Iy*integral_y
 
        valx=round(valx,0)
        valy=round(valy,0) # nearest whole number
        
        # scale values to a range of 100, then put on range of -50 to 50

        valx_scaled = (float(valx - 0) / float(100)) - 50
        valy_scaled = (float(valy - 0) / float(100)) - 50

        # Z axis movement
        pf_b = 0
        # X axis movement
        pr_l = valx_scaled
        # Y axis movement
        pu_d = valy_scaled

    return pf_b,pr_l,pu_d

#############################################################################################
#                                    Object detection
#############################################################################################

def make_red_dots():
    ret, frame = cap.read()
    inp = cv2.resize(frame, (w , h))
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    boxes, scores, classes, num_detections = detector(rgb_tensor)
    pred_labels = classes.numpy().astype('int')[0]
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]
    x_dots = []
    y_dots = []
    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue  
        
        #likelyness
        #score_txt = f'{100 * round(score,0)}'

        # Red Dot: find center and draw a red dot
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        img = cv2.circle(img, (x,y), radius=1, color=(255, 0, 0), thickness=-1)
        x_dots.append(x)
        y_dots.append(y)

    #save computational power
    #cv2.imshow('Result',img)
    return x_dots,y_dots


#############################################################################################
#                                    Serial Communication
#############################################################################################

def main():
    while True:
        x_dots,y_dots = make_red_dots()
        px,py = priority_coordinates(x_dots,y_dots)
        pf_b,pr_l,pu_d = coordinates_2_power(px,py)
        data = str("{0:.2f}".format(pf_b)) + "," + str("{0:.2f}".format(pr_l)) + "," + str("{0:.2f}".format(pu_d))
        print(data)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    '''    
    with serial.Serial(serial_device, baudrate, timeout=timeout) as arduino:
        try:
            while True:
                x_dots,y_dots = make_red_dots()
                px,py = priority_coordinates(x_dots,y_dots)
                pf_b,pr_l,pu_d = coordinates_2_power(px,py)
                data = str("{0:.2f}".format(pf_b)) + "," + str("{0:.2f}".format(pr_l)) + "," + str("{0:.2f}".format(pu_d))
                arduino.write(data.encode())
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("KeyboardInterrupt has been caught.")
    '''
    cap.release()
    cv2.destroyAllWindows()

main()