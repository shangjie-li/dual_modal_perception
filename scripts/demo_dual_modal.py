# -*- coding: UTF-8 -*-

import os
import sys
import rospy
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import threading
import argparse

from std_msgs.msg import Header
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from yolov5_detector import Yolov5Detector, draw_predictions
from mono_estimator import MonoEstimator
from functions import get_stamp, publish_image
from functions import display, print_info
from functions import simplified_nms

parser = argparse.ArgumentParser(
    description='Demo script for dual modal peception')
parser.add_argument('--print', action='store_true',
    help='Whether to print and record infos.')
parser.add_argument('--sub_image1', default='/pub_rgb', type=str,
    help='The image topic to subscribe.')
parser.add_argument('--sub_image2', default='/pub_t', type=str,
    help='The image topic to subscribe.')
parser.add_argument('--pub_image', default='/result', type=str,
    help='The image topic to publish.')
parser.add_argument('--calib_file', default='../conf/calibration_image.yaml', type=str,
    help='The calibration file of the camera.')
parser.add_argument('--modality', default='RGBT', type=str,
    help='The modality to use. This should be `RGB`, `T` or `RGBT`.')
parser.add_argument('--indoor', action='store_true',
    help='Whether to use INDOOR detection mode.')
parser.add_argument('--frame_rate', default=10, type=int,
    help='Working frequency.')
parser.add_argument('--display', action='store_true',
    help='Whether to display and save all videos.')
args = parser.parse_args()

image1_lock = threading.Lock()
image2_lock = threading.Lock()

def image1_callback(image):
    global image1_stamp, image1_frame
    image1_lock.acquire()
    image1_stamp = get_stamp(image.header)
    image1_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image1_lock.release()

def image2_callback(image):
    global image2_stamp, image2_frame
    image2_lock.acquire()
    image2_stamp = get_stamp(image.header)
    image2_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image2_lock.release()

def timer_callback(event):
    global image1_stamp, image1_frame
    image1_lock.acquire()
    cur_stamp1 = image1_stamp
    cur_frame1 = image1_frame.copy()
    image1_lock.release()
    
    global image2_stamp, image2_frame
    image2_lock.acquire()
    cur_stamp2 = image2_stamp
    cur_frame2 = image2_frame.copy()
    image2_lock.release()
    
    global frame
    frame += 1
    start = time.time()
    if args.indoor:
        labels, scores, boxes = detector.run(
            cur_frame1, conf_thres=0.50, classes=[0]
        ) # person
    else:
        if args.modality.lower() == 'rgb':
            labels, scores, boxes = detector1.run(
                cur_frame1, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
        elif args.modality.lower() == 't':
            labels, scores, boxes = detector2.run(
                cur_frame2, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
        elif args.modality.lower() == 'rgbt':
            labels1, scores1, boxes1 = detector1.run(
                cur_frame1, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
            labels2, scores2, boxes2 = detector2.run(
                cur_frame2, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
            ) # pedestrian, cyclist, car, bus, truck
            labels = labels1 + labels2
            scores = scores1 + scores2
            if boxes1.shape[0] > 0 and boxes2.shape[0] > 0:
                boxes = np.concatenate([boxes1, boxes2], axis=0)
                indices = simplified_nms(boxes, scores)
                labels, scores, boxes = np.array(labels)[indices], np.array(scores)[indices], boxes[indices]
            elif boxes1.shape[0] > 0:
                boxes = boxes1
            elif boxes2.shape[0] > 0:
                boxes = boxes2
            else:
                boxes = np.array([])
        else:
            raise ValueError("The modality must be 'RGB', 'T' or 'RGBT'.")
    labels_temp = labels.copy()
    labels = []
    for i in labels_temp:
        labels.append(i if i not in ['pedestrian', 'cyclist'] else 'person')
    
    locations = mono.estimate(boxes)
    indices = [i for i in range(len(locations)) if locations[i][1] > 0 and locations[i][1] < 200]
    labels, scores, boxes, locations = \
        np.array(labels)[indices], np.array(scores)[indices], boxes[indices], np.array(locations)[indices]
    distances = [(loc[0] ** 2 + loc[1] ** 2) ** 0.5 for loc in locations]
    cur_frame1 = cur_frame1[:, :, ::-1].copy() # to BGR
    cur_frame2 = cur_frame2[:, :, ::-1].copy() # to BGR
    for i in reversed(np.argsort(distances)):
        cur_frame1 = draw_predictions(
            cur_frame1, str(labels[i]), float(scores[i]), boxes[i], location=locations[i]
        )
        cur_frame2 = draw_predictions(
            cur_frame2, str(labels[i]), float(scores[i]), boxes[i], location=locations[i]
        )
    result_frame = np.concatenate([cur_frame1, cur_frame2], axis=1)
    
    if args.display:
        if not display(result_frame, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")
    result_frame = result_frame[:, :, ::-1] # to RGB
    publish_image(pub, result_frame)
    delay = round(time.time() - start, 3)
    
    if args.print:
        print_info(frame, cur_stamp1, delay, labels, scores, boxes, locations, file_name)

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("dual_modal_perception", anonymous=True, disable_signals=True)
    frame = 0
    
    # 记录时间戳和检测结果
    if args.print:
        file_name = 'result.txt'
        with open(file_name, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    # 设置标定参数
    if not os.path.exists(args.calib_file):
        raise ValueError("%s Not Found" % (args.calib_file))
    mono = MonoEstimator(args.calib_file, print_info=args.print)
    
    # 初始化Yolov5Detector
    if args.indoor:
        detector = Yolov5Detector(weights='weights/coco/yolov5s.pt')
    else:
        if args.modality.lower() == 'rgb':
            detector1 = Yolov5Detector(weights='weights/seumm_visible/yolov5s_50ep_pretrained.pt')
        elif args.modality.lower() == 't':
            detector2 = Yolov5Detector(weights='weights/seumm_lwir/yolov5s_100ep_pretrained.pt')
        elif args.modality.lower() == 'rgbt':
            detector1 = Yolov5Detector(weights='weights/seumm_visible/yolov5s_50ep_pretrained.pt')
            detector2 = Yolov5Detector(weights='weights/seumm_lwir/yolov5s_100ep_pretrained.pt')
        else:
            raise ValueError("The modality must be 'RGB', 'T' or 'RGBT'.")
    
    # 准备图像序列
    image1_stamp, image1_frame = None, None
    image2_stamp, image2_frame = None, None
    rospy.Subscriber(args.sub_image1, Image, image1_callback, queue_size=1,
        buff_size=52428800)
    rospy.Subscriber(args.sub_image2, Image, image2_callback, queue_size=1,
        buff_size=52428800)
    while image1_frame is None or image2_frame is None:
        time.sleep(0.1)
        print('Waiting for topic %s and %s...' % (args.sub_image1, args.sub_image2))
    print('  Done.\n')
    
    # 保存视频
    if args.display:
        assert image1_frame.shape == image2_frame.shape, \
            'image1_frame.shape must be equal to image2_frame.shape.'
        win_h, win_w = image1_frame.shape[0], image1_frame.shape[1] * 2
        v_path = 'result.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_writer = cv2.VideoWriter(v_path, v_format, args.frame_rate, (win_w, win_h), True)
    
    # 启动定时检测线程
    pub = rospy.Publisher(args.pub_image, Image, queue_size=1)
    rospy.Timer(rospy.Duration(1 / args.frame_rate), timer_callback)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
