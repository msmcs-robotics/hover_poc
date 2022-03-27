import sys
import time
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# change to increase/decrease performance
h = 300
w = 300

MODEL_PATH = './model/frozen_inference_graph_face.pb'
LABELS_FILE = './protos/face_label_map.pbtxt'
NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(LABELS_FILE)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

device = "/dev/video0"
#device = 0
cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, h)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, w)
cap.set(cv2.CAP_PROP_FPS, 10) # force to 10fps to save on CPU

class TensoflowFaceDector(object):

    def __init__(self, MODEL_PATH):

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
       
        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        return (boxes, scores, classes, num_detections)

t_detect = TensoflowFaceDector(MODEL_PATH)

while True:
        ret, image = cap.read()
        if ret == 0:
            break

        image = cv2.flip(image, 1)

        (boxes, scores, classes, num_detections) = t_detect.run(image)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)


        cv2.imshow('',image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

cap.release()