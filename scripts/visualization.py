import rospy

import cv2
import math
import numpy as np
from fit_3d_model import transform_2d_point_clouds, compute_obb_vertex_coordinates
from get_ellipse import get_ellipse

def get_stamp(header):
    return header.stamp.secs + 0.000000001 * header.stamp.nsecs

def normalize_phi(phi):
    while phi < 0:
        phi += math.pi
    while phi >= math.pi:
        phi -= math.pi
    return phi

def draw_object_info(img, classname, number, xref, yref, vx, vy, uv_1, uv_2, color,
    display_state=True, l=0, w=0, h=0, phi=0):
    # 功能：在图像上绘制目标信息
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      classname <class 'str'> 类别
    #      number <class 'int'> 编号
    #      xref <class 'float'> X坐标
    #      yref <class 'float'> Y坐标
    #      vx <class 'float'> X方向速度
    #      vy <class 'float'> Y方向速度
    #      uv_1 <class 'numpy.ndarray'> (1, 2) 代表在图像坐标[u, v]处绘制class和id
    #      uv_2 <class 'numpy.ndarray'> (1, 2) 代表在图像坐标[u, v]处绘制state
    #      color <class 'tuple'> RGB颜色
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    u1, v1, u2, v2 = int(uv_1[0, 0]), int(uv_1[0, 1]), int(uv_2[0, 0]), int(uv_2[0, 1])
    
    f_face = cv2.FONT_HERSHEY_DUPLEX
    f_scale = 1.0
    f_thickness = 1
    white = (255, 255, 255)
    
    text_str = classname + ' ' + str(number)
    text_w, text_h = cv2.getTextSize(text_str, f_face, f_scale, f_thickness)[0]
    cv2.rectangle(img, (u1, v1), (u1 + text_w, v1 - text_h - 4), color, -1)
    cv2.putText(img, text_str, (u1, v1 - 3), f_face, f_scale, white, f_thickness, cv2.LINE_AA)
    
    if display_state:
        text_lo = '(%.1fm, %.1fm)' % (xref, yref)
        text_w_lo, _ = cv2.getTextSize(text_lo, f_face, f_scale, f_thickness)[0]
        cv2.rectangle(img, (u2, v2), (u2 + text_w_lo, v2 + text_h + 4), color, -1)
        cv2.putText(img, text_lo, (u2, v2 + text_h + 1), f_face, f_scale, white, f_thickness, cv2.LINE_AA)
        
        text_ve = '(%.1fm/s, %.1fm/s)' % (vx, vy)
        text_w_ve, _ = cv2.getTextSize(text_ve, f_face, f_scale, f_thickness)[0]
        cv2.rectangle(img, (u2, v2 + text_h + 4), (u2 + text_w_ve, v2 + 2 * text_h + 8), color, -1)
        cv2.putText(img, text_ve, (u2, v2 + 2 * text_h + 5), f_face, f_scale, white, f_thickness, cv2.LINE_AA)
        
        text_sc = '(%.1fm, %.1fm, %.1fm, %.1frad)' % (l, w, h, phi)
        text_w_sc, _ = cv2.getTextSize(text_sc, f_face, f_scale, f_thickness)[0]
        cv2.rectangle(img, (u2, v2 + 2 * text_h + 8), (u2 + text_w_sc, v2 + 3 * text_h + 12), color, -1)
        cv2.putText(img, text_sc, (u2, v2 + 3 * text_h + 9), f_face, f_scale, white, f_thickness, cv2.LINE_AA)
    
    return img

def draw_object_model_from_main_view(img, objs, frame, display_frame=True,
    display_state=True, thickness=2):
    # 功能：在图像上绘制目标轮廓
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      objs <class 'list'> 存储目标检测结果
    #      frame <class 'int'> 帧数
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    # 按距离从远到近进行排序
    num = len(objs)
    dds = []
    for i in range(num):
        dd = objs[i].x0 ** 2 + objs[i].y0 ** 2
        dds.append(dd)
    idxs = list(np.argsort(dds))
    
    for i in reversed(idxs):
        # 绘制二维边界框
        xyxy = objs[i].box
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img, c1, c2, objs[i].color, thickness)
        
        # 选取二维边界框中的顶点，绘制目标信息
        uv_1 = np.array([int(xyxy[0]), int(xyxy[1])]).reshape(1, 2)
        uv_2 = np.array([int(xyxy[0]), int(xyxy[3])]).reshape(1, 2)
        
        img = draw_object_info(img, objs[i].classname, objs[i].number,
            objs[i].x0, objs[i].y0, objs[i].vx, objs[i].vy, uv_1, uv_2, objs[i].color,
            display_state, objs[i].l, objs[i].w, objs[i].h, objs[i].phi)
    
    f_face = cv2.FONT_HERSHEY_DUPLEX
    f_scale = 1.0
    f_thickness = 1
    red = (0, 0, 255)
    if display_frame:
        cv2.putText(img, str(frame), (20, 40), f_face, f_scale, red, f_thickness, cv2.LINE_AA)
    
    return img

def draw_object_model_from_bev_view(img, objs, display_gate=True,
    center_alignment=True, axis=True, thickness=2):
    # 功能：在图像上绘制目标轮廓
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      objs <class 'list'> 存储目标检测结果
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    num = len(objs)
    for i in range(num):
        height, width = img.shape[0], img.shape[1]
        h2, w2 = height // 2, width // 2
        if axis:
            black = (0, 0, 0)
            if center_alignment:
                cv2.line(img, (w2, h2), (w2 + 10, h2), black, thickness)
                cv2.line(img, (w2, h2), (w2, h2 - 10), black, thickness)
            else:
                cv2.line(img, (0, h2), (10, h2), black, thickness)
                cv2.line(img, (0, h2), (0, h2 - 10), black, thickness)
        
        xst, yst = compute_obb_vertex_coordinates(objs[i].x0, objs[i].y0, objs[i].l, objs[i].w, objs[i].phi)
        xst, yst = (xst * 10).astype(int), (- yst * 10).astype(int)
        if center_alignment:
            xst, yst = transform_2d_point_clouds(xst, yst, 0, width / 2, height / 2)
        else:
            xst, yst = transform_2d_point_clouds(xst, yst, 0, 0, height / 2)
        for j in range(4):
            pt_1 = (int(xst[j]), int(yst[j]))
            pt_2 = (int(xst[(j + 1) % 4]), int(yst[(j + 1) % 4]))
            cv2.line(img, pt_1, pt_2, objs[i].color, thickness)
        
        if display_gate:
            # 绘制跟踪门椭圆
            a, b = objs[i].tracker.compute_association_gate(objs[i].tracker.gate_threshold)
            x, y = objs[i].x0, objs[i].y0
            xs, ys = get_ellipse(x, y, a, b, 0)
            xs, ys = np.array(xs), np.array(ys)
            
            img = draw_point_clouds_from_bev_view(img, xs, ys, center_alignment, objs[i].color, radius=1)
        
    return img
