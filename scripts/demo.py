# -*- coding: UTF-8 -*-

import os
import sys
import rospy
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import threading

from std_msgs.msg import Header
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    
from config import Object
from config import BasicColor

from mono_estimator import MonoEstimator
from kalman_filter import KalmanFilter4D

from visualization import get_stamp
from visualization import normalize_phi
from visualization import draw_object_model_from_main_view
from visualization import draw_object_model_from_bev_view

image1_lock = threading.Lock()
image2_lock = threading.Lock()

def estimate_by_monocular(mono, masks, classes, scores, boxes):
    objs = []
    num = classes.shape[0] if classes is not None else 0
    if num > 0:
        u, v = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]
    
    for i in range(num):
        if v[i] > win_h / 2:
            obj = Object()
            obj.mask = masks[i].clone() # clone for tensor
            obj.classname = classes[i]
            obj.score = scores[i]
            obj.box = boxes[i].copy() # copy for ndarray
            
            x, y, z = mono.uv_to_xyz(u[i], v[i])
            obj.x0 = z
            obj.y0 = -x
            
            obj.l = 2.0
            obj.w = 2.0
            obj.h = 2.0
            
            obj.color = BasicColor
            objs.append(obj)
        
    return objs

def track(number, objs_tracked, objs_temp, objs_detected, blind_update_limit, frame_rate):
    # 数据关联与跟踪
    num = len(objs_tracked)
    for j in range(num):
        # 计算残差加权范数
        flag, idx, ddm = False, 0, float('inf')
        n = len(objs_detected)
        for k in range(n):
            zx = objs_detected[k].x0
            zy = objs_detected[k].y0
            dd = objs_tracked[j].tracker.compute_the_residual(zx, zy)
            if dd < ddm and dd < objs_tracked[j].tracker.gate_threshold:
                flag, idx, ddm = True, k, dd
        
        if flag and objs_tracked[j].classname == objs_detected[idx].classname:
            # 匹配成功，预测并更新
            objs_tracked[j].tracker.predict()
            objs_tracked[j].tracker.update(objs_detected[idx].x0, objs_detected[idx].y0)
            
            # 继承检测结果中的参数
            objs_tracked[j].mask = objs_detected[idx].mask.clone()
            objs_tracked[j].score = objs_detected[idx].score
            objs_tracked[j].box = objs_detected[idx].box.copy()
            objs_tracked[j].l = objs_detected[idx].l
            objs_tracked[j].w = objs_detected[idx].w
            objs_tracked[j].h = objs_detected[idx].h
            objs_tracked[j].phi = objs_detected[idx].phi
            objs_tracked[j].color = objs_detected[idx].color
            objs_tracked[j].refined_by_lidar = objs_detected[idx].refined_by_lidar
            objs_tracked[j].refined_by_radar = objs_detected[idx].refined_by_radar
            
            # 修改更新中断次数
            objs_tracked[j].tracker_blind_update = 0
            objs_detected.pop(idx)
        else:
            # 匹配不成功，只预测
            objs_tracked[j].tracker.predict()
            
            # 修改更新中断次数
            objs_tracked[j].tracker_blind_update += 1
        
        # 输出滤波值
        objs_tracked[j].x0 = objs_tracked[j].tracker.xx[0, 0]
        objs_tracked[j].vx = objs_tracked[j].tracker.xx[1, 0]
        objs_tracked[j].y0 = objs_tracked[j].tracker.xx[2, 0]
        objs_tracked[j].vy = objs_tracked[j].tracker.xx[3, 0]
        
    # 删除长时间未跟踪的目标
    objs_remained = []
    num = len(objs_tracked)
    for j in range(num):
        if objs_tracked[j].tracker_blind_update <= blind_update_limit:
            objs_remained.append(objs_tracked[j])
    objs_tracked = objs_remained
    
    # 增广跟踪列表
    num = len(objs_temp)
    for j in range(num):
        # 计算残差加权范数
        flag, idx, ddm = False, 0, float('inf')
        n = len(objs_detected)
        for k in range(n):
            zx = objs_detected[k].x0
            zy = objs_detected[k].y0
            dd = objs_temp[j].tracker.compute_the_residual(zx, zy)
            if dd < ddm and dd < objs_temp[j].tracker.gate_threshold:
                flag, idx, ddm = True, k, dd
        
        if flag and objs_temp[j].classname == objs_detected[idx].classname:
            zx = objs_detected[idx].x0
            zy = objs_detected[idx].y0
            x = objs_temp[j].tracker.xx[0, 0]
            y = objs_temp[j].tracker.xx[2, 0]
            vx = (zx - x) / objs_temp[j].tracker.ti
            vy = (zy - y) / objs_temp[j].tracker.ti
            
            # 继承检测结果中的参数
            objs_temp[j].mask = objs_detected[idx].mask.clone()
            objs_temp[j].score = objs_detected[idx].score
            objs_temp[j].box = objs_detected[idx].box.copy()
            objs_temp[j].l = objs_detected[idx].l
            objs_temp[j].w = objs_detected[idx].w
            objs_temp[j].h = objs_detected[idx].h
            objs_temp[j].phi = objs_detected[idx].phi
            objs_temp[j].color = objs_detected[idx].color
            objs_temp[j].refined_by_lidar = objs_detected[idx].refined_by_lidar
            objs_temp[j].refined_by_radar = objs_detected[idx].refined_by_radar
            
            # 对跟踪的位置、速度重新赋值
            objs_temp[j].x0 = objs_temp[j].tracker.xx[0, 0] = zx
            objs_temp[j].vx = objs_temp[j].tracker.xx[1, 0] = vx
            objs_temp[j].y0 = objs_temp[j].tracker.xx[2, 0] = zy
            objs_temp[j].vy = objs_temp[j].tracker.xx[3, 0] = vy
            
            # 增加ID
            number += 1
            objs_temp[j].number = number
            objs_tracked.append(objs_temp[j])
            objs_detected.pop(idx)
    
    # 增广临时跟踪列表
    objs_temp = objs_detected
    num = len(objs_temp)
    for j in range(num):
        # 初始化卡尔曼滤波器，对目标进行跟踪
        objs_temp[j].tracker = KalmanFilter4D(
            1 / frame_rate,
            objs_temp[j].x0, objs_temp[j].vx, objs_temp[j].y0, objs_temp[j].vy,
            sigma_ax=1, sigma_ay=1, sigma_ox=0.1, sigma_oy=0.1, gate_threshold=400
        )
    
    return number, objs_tracked, objs_temp

def image1_callback(image):
    global image1_stamps, image1_frames
    stamp = get_stamp(image.header)
    data = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    
    image1_lock.acquire()
    if len(image1_stamps) < 30:
        image1_stamps.append(stamp)
        image1_frames.append(data)
    else:
        image1_stamps.pop(0)
        image1_frames.pop(0)
        image1_stamps.append(stamp)
        image1_frames.append(data)
    image1_lock.release()

def image2_callback(image):
    global image2_stamps, image2_frames
    stamp = get_stamp(image.header)
    data = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    
    image2_lock.acquire()
    if len(image2_stamps) < 30:
        image2_stamps.append(stamp)
        image2_frames.append(data)
    else:
        image2_stamps.pop(0)
        image2_frames.pop(0)
        image2_stamps.append(stamp)
        image2_frames.append(data)
    image2_lock.release()

def fusion_callback(event):
    time_all_start = time.time()
    global objs_tracked, objs_temp, number, frame
    frame += 1
    
    global image1_stamps, image1_frames
    global image2_stamps, image2_frames
    
    image1_lock.acquire()
    image1_stamp = image1_stamps[-1]
    image1_frame = image1_frames[-1].copy()
    image1_lock.release()
    
    image2_lock.acquire()
    image2_stamp = image2_stamps[-1]
    image2_frame = image2_frames[-1].copy()
    image2_lock.release()
    
    image1_frame = image1_frame[:, :, ::-1] # to BGR
    image2_frame = image2_frame[:, :, ::-1] # to BGR
    current_image1 = image1_frame.copy()
    current_image2 = image2_frame.copy()
    
    # 实例分割与视觉估计
    time_segmentation_start = time.time()
    if modality == 'RGB':
        masks, classes, scores, boxes = detector.run(image1_frame)
    elif modality == 'T':
        masks, classes, scores, boxes = detector.run(image2_frame)
    elif modality == 'RGBT':
        masks, classes, scores, boxes = detector.run(image1_frame, image2_frame)
    else:
        raise ValueError("modality must be 'RGB', 'T' or 'RGBT'.")
    
    objs = estimate_by_monocular(mono, masks, classes, scores, boxes)
    time_segmentation = time.time() - time_segmentation_start
    
    # 目标跟踪
    time_tracking_start = time.time()
    if processing_mode == 'DT':
        number, objs_tracked, objs_temp = track(number, objs_tracked, objs_temp, objs,
            blind_update_limit, frame_rate)
        if number >= max_id:
            number = 0
    time_tracking = time.time() - time_tracking_start
    
    if processing_mode == 'D':
        objs = objs
    elif processing_mode == 'DT':
        objs = objs_tracked
    else:
        raise ValueError("processing_mode must be 'D' or 'DT'.")
    
    # 可视化
    time_display_start = time.time()
    if display_image_raw:
        win_image1_raw = current_image1.copy()
        win_image2_raw = current_image2.copy()
    if display_image_segmented:
        win_image1_segmented = current_image1.copy()
        win_image2_segmented = current_image2.copy()
        num = len(objs)
        for i in range(num):
            if objs[i].tracker_blind_update > 0:
                continue
            win_image1_segmented = draw_segmentation_result(
                win_image1_segmented, objs[i].mask, objs[i].classname, objs[i].score,
                objs[i].box, BasicColor
            )
            win_image2_segmented = draw_segmentation_result(
                win_image2_segmented, objs[i].mask, objs[i].classname, objs[i].score,
                objs[i].box, BasicColor
            )
    if display_2d_modeling:
        win_image1_2d_modeling = np.ones((480, 640, 3), dtype=np.uint8) * 255
        win_image2_2d_modeling = np.ones((480, 640, 3), dtype=np.uint8) * 255
        win_image1_2d_modeling = draw_object_model_from_bev_view(
            win_image1_2d_modeling, objs, display_gate,
            center_alignment=False, axis=True, thickness=2
        )
        win_image2_2d_modeling = draw_object_model_from_bev_view(
            win_image2_2d_modeling, objs, display_gate,
            center_alignment=False, axis=True, thickness=2
        )
        win_image1_2d_modeling = cv2.resize(win_image1_2d_modeling, (win_w, win_h))
        win_image2_2d_modeling = cv2.resize(win_image2_2d_modeling, (win_w, win_h))
    if display_3d_modeling:
        win_image1_3d_modeling = current_image1.copy()
        win_image2_3d_modeling = current_image2.copy()
        win_image1_3d_modeling = draw_object_model_from_main_view(
            win_image1_3d_modeling, objs, frame, display_frame,
            display_obj_state, thickness=2
        )
        win_image2_3d_modeling = draw_object_model_from_main_view(
            win_image2_3d_modeling, objs, frame, display_frame,
            display_obj_state, thickness=2
        )
    
    # 显示与保存
    if display_image_raw:
        cv2.namedWindow("image1_raw", cv2.WINDOW_NORMAL)
        cv2.imshow("image1_raw", win_image1_raw)
        v_image1_raw.write(win_image1_raw)
        cv2.namedWindow("image2_raw", cv2.WINDOW_NORMAL)
        cv2.imshow("image2_raw", win_image2_raw)
        v_image2_raw.write(win_image2_raw)
    if display_image_segmented:
        cv2.namedWindow("image1_segmented", cv2.WINDOW_NORMAL)
        cv2.imshow("image1_segmented", win_image1_segmented)
        v_image1_segmented.write(win_image1_segmented)
        cv2.namedWindow("image2_segmented", cv2.WINDOW_NORMAL)
        cv2.imshow("image2_segmented", win_image2_segmented)
        v_image2_segmented.write(win_image2_segmented)
    if display_2d_modeling:
        cv2.namedWindow("image1_2d_modeling", cv2.WINDOW_NORMAL)
        cv2.imshow("image1_2d_modeling", win_image1_2d_modeling)
        v_image1_2d_modeling.write(win_image1_2d_modeling)
        cv2.namedWindow("image2_2d_modeling", cv2.WINDOW_NORMAL)
        cv2.imshow("image2_2d_modeling", win_image2_2d_modeling)
        v_image2_2d_modeling.write(win_image2_2d_modeling)
    if display_3d_modeling:
        cv2.namedWindow("image1_3d_modeling", cv2.WINDOW_NORMAL)
        cv2.imshow("image1_3d_modeling", win_image1_3d_modeling)
        v_image1_3d_modeling.write(win_image1_3d_modeling)
        cv2.namedWindow("image2_3d_modeling", cv2.WINDOW_NORMAL)
        cv2.imshow("image2_3d_modeling", win_image2_3d_modeling)
        v_image2_3d_modeling.write(win_image2_3d_modeling)
    
    # 显示窗口时按Esc键终止程序
    display = [display_image_raw, display_image_segmented,
        display_2d_modeling, display_3d_modeling].count(True) > 0
    if display and cv2.waitKey(1) == 27:
        if display_image_raw:
            cv2.destroyWindow("image1_raw")
            cv2.destroyWindow("image2_raw")
            v_image1_raw.release()
            v_image2_raw.release()
            print("Save video of image_raw.")
        if display_image_segmented:
            cv2.destroyWindow("image1_segmented")
            cv2.destroyWindow("image2_segmented")
            v_image1_segmented.release()
            v_image2_segmented.release()
            print("Save video of image_segmented.")
        if display_2d_modeling:
            cv2.destroyWindow("image1_2d_modeling")
            cv2.destroyWindow("image2_2d_modeling")
            v_image1_2d_modeling.release()
            v_image2_2d_modeling.release()
            print("Save video of 2d_modeling.")
        if display_3d_modeling:
            cv2.destroyWindow("image1_3d_modeling")
            cv2.destroyWindow("image2_3d_modeling")
            v_image1_3d_modeling.release()
            v_image2_3d_modeling.release()
            print("Save video of 3d_modeling.")
        print("\nReceived the shutdown signal.\n")
        rospy.signal_shutdown("Everything is over now.")
    time_display = time.time() - time_display_start
    
    # 记录耗时情况
    time_segmentation = round(time_segmentation, 3)
    time_tracking = round(time_tracking, 3) if processing_mode == 'DT' else None
    time_display = round(time_display, 3) if display else None
    time_all = round(time.time() - time_all_start, 3)
    
    if print_time:
        image_stamp = image1_stamp
        print("image_stamp          ", image_stamp)
        print("time of segmentation ", time_segmentation)
        print("time of tracking     ", time_tracking)
        print("time of display      ", time_display)
        print("time of all          ", time_all)
        print()
    
    if print_objects_info:
        num = len(objs)
        for j in range(num):
            n = objs[j].number if processing_mode == 'DT' else j
            print('Object %d x0:%.2f y0:%.2f %s' % (
                n, objs[j].x0, objs[j].y0, objs[j].classname))
        print()
    
    if record_time:
        time_tracking = 0.0 if time_tracking is None else time_tracking
        time_display = 0.0 if time_display is None else time_display
        with open(filename_time, 'a') as fob:
            content = 'frame:%d amount:%d segmentation:%.3f ' + \
                'tracking:%.3f display:%.3f all:%.3f'
            fob.write(
                content % (
                frame, len(objs), time_segmentation,
                time_tracking, time_display, time_all)
            )
            fob.write('\n')
            
    if record_objects_info:
        num = len(objs)
        for j in range(num):
            n = objs[j].number if processing_mode == 'DT' else j
            with open(filename_objects_info, 'a') as fob:
                content = 'stamp:%.3f frame:%d id:%d ' + \
                    'xref:%.3f vx:%.3f yref:%.3f vy:%.3f ' + \
                    'x0:%.3f y0:%.3f z0:%.3f l:%.3f w:%.3f h:%.3f phi:%.3f'
                fob.write(
                    content % (
                    image_stamp, frame, n,
                    objs[j].x0, objs[j].vx, objs[j].y0, objs[j].vy,
                    objs[j].x0, objs[j].y0, objs[j].z0,
                    objs[j].l, objs[j].w, objs[j].h, objs[j].phi)
                )
                fob.write('\n')

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("dt")
    
    # 记录耗时
    print_time = rospy.get_param("~print_time")
    record_time = rospy.get_param("~record_time")
    if record_time:
        filename_time = 'time_cost.txt'
        with open(filename_time, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    # 记录结果
    print_objects_info = rospy.get_param("~print_objects_info")
    record_objects_info = rospy.get_param("~record_objects_info")
    if record_objects_info:
        filename_objects_info = 'result.txt'
        with open(filename_objects_info, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    # 设置ROS消息名称
    sub_image1_topic = rospy.get_param("~sub_image1_topic")
    sub_image2_topic = rospy.get_param("~sub_image2_topic")
    
    # 设置标定参数
    cwd = os.getcwd()
    calib_pth_idx = -1
    while cwd[calib_pth_idx] != '/':
        calib_pth_idx -= 1
    
    calibration_image_file = rospy.get_param("~calibration_image_file")
    f_path = cwd[:calib_pth_idx] + '/conf/' + calibration_image_file
    if not os.path.exists(f_path):
            raise ValueError("%s doesn't exist." % (f_path))
    mono = MonoEstimator(f_path, print_info=True)
    
    # 初始化YolactDetector
    modality = rospy.get_param("~modality")
    if modality == 'RGB':
        from yolact_detector import YolactDetector, draw_segmentation_result
        detector = YolactDetector()
    elif modality == 'T':
        from yolact_detector import YolactDetector, draw_segmentation_result
        detector = YolactDetector()
    elif modality == 'RGBT':
        from dual_modal_yolact_detector import YolactDetector, draw_segmentation_result
        detector = YolactDetector()
    else:
        raise ValueError("modality must be 'RGB', 'T' or 'RGBT'.")
    
    # 准备图像序列
    print('Waiting for topic...')
    image1_stamps, image1_frames = [], []
    image2_stamps, image2_frames = [], []
    rospy.Subscriber(sub_image1_topic, Image, image1_callback, queue_size=1,
        buff_size=52428800)
    rospy.Subscriber(sub_image2_topic, Image, image2_callback, queue_size=1,
        buff_size=52428800)
    while len(image1_stamps) < 30 or len(image2_stamps) < 30:
        time.sleep(1)
        print('Waiting for topic %s and %s...' % (sub_image1_topic, sub_image2_topic))
    
    # 完成所需传感器数据序列
    print('  Done.\n')
    
    # 初始化检测列表、跟踪列表、临时跟踪列表
    objs_tracked = []
    objs_temp = []
    number = 0
    frame = 0
    
    # 设置更新中断次数的限制、工作帧率、最大ID
    blind_update_limit = rospy.get_param("~blind_update_limit")
    frame_rate = rospy.get_param("~frame_rate")
    max_id = rospy.get_param("~max_id")
    
    # 设置显示窗口
    display_image_raw = rospy.get_param("~display_image_raw")
    display_image_segmented = rospy.get_param("~display_image_segmented")
    
    display_2d_modeling = rospy.get_param("~display_2d_modeling")
    display_gate = rospy.get_param("~display_gate")
    
    display_3d_modeling = rospy.get_param("~display_3d_modeling")
    display_frame = rospy.get_param("~display_frame")
    display_obj_state = rospy.get_param("~display_obj_state")
    
    win_h1, win_w1 = image1_frames[0].shape[0], image1_frames[0].shape[1]
    win_h2, win_w2 = image2_frames[0].shape[0], image2_frames[0].shape[1]
    assert win_h1 == win_h2 and win_w1 == win_w2, \
        '(win_h1, win_w1) should be the same as (win_h2, win_w2).'
    win_h = win_h1
    win_w = win_w1
    
    if display_image_raw:
        v_path = 'image1_raw.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image1_raw = cv2.VideoWriter(v_path, v_format, frame_rate,
            (win_w, win_h), True)
        v_path = 'image2_raw.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image2_raw = cv2.VideoWriter(v_path, v_format, frame_rate,
            (win_w, win_h), True)
    if display_image_segmented:
        v_path = 'image1_segmented.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image1_segmented = cv2.VideoWriter(v_path, v_format, frame_rate,
            (win_w, win_h), True)
        v_path = 'image2_segmented.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image2_segmented = cv2.VideoWriter(v_path, v_format, frame_rate,
            (win_w, win_h), True)
    if display_2d_modeling:
        v_path = 'image1_2d_modeling.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image1_2d_modeling = cv2.VideoWriter(v_path, v_format, frame_rate,
            (win_w, win_h), True)
        v_path = 'image2_2d_modeling.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image2_2d_modeling = cv2.VideoWriter(v_path, v_format, frame_rate,
            (win_w, win_h), True)
    if display_3d_modeling:
        v_path = 'image1_3d_modeling.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image1_3d_modeling = cv2.VideoWriter(v_path, v_format, frame_rate,
            (win_w, win_h), True)
        v_path = 'image2_3d_modeling.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_image2_3d_modeling = cv2.VideoWriter(v_path, v_format, frame_rate,
            (win_w, win_h), True)
    
    # 启动数据融合线程
    processing_mode = rospy.get_param("~processing_mode")
    if processing_mode == 'D':
        display_gate = False
    elif processing_mode == 'DT':
        display_gate = display_gate
    else:
        raise ValueError("processing_mode must be 'D' or 'DT'.")
        
    rospy.Timer(rospy.Duration(1 / frame_rate), fusion_callback)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
