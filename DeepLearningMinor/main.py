import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from math_convert import *
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

cap = cv2.VideoCapture(0)

sys.path.append("..")

MODEL_NAME = 'model'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
NUM_CLASSES = 24

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            ret, frame = cap.read()
            height = 720
            width = 1280
            image_np_expanded = np.expand_dims(frame, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5)

            od_list = [[category_index.get(value).get('name'), boxes[0][index][1] * width] for index,
                   value in enumerate(classes[0]) if scores[0, index] > 0.65]
            od_list_seq = sorted(od_list, key=lambda x:(-x[1], x[0]), reverse=True)
            od_list_co = [seq[0] for seq in od_list_seq]
            od_list_co = convop(od_list_co)
            co_num_list = combint(od_list_co)
            exp_result = chkfl(co_num_list)
            result = getresult(co_num_list, exp_result)

            if str(result) == '...':
                obj = str(exp_result)
            else:
                obj = str(exp_result) + ' is ' + str(result)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, obj, (0, 430), font, 2, (0, 0, 0), 0, cv2.LINE_AA)
            cv2.imshow('object detection', cv2.resize(frame, (width, height)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                # videoFile.release()
                cv2.destroyAllWindows()
                break
