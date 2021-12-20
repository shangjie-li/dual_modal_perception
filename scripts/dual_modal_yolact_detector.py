# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt # For WARNING: QApplication was not created in the main() thread.

import os
import sys
cwd = os.getcwd()
idx = -1
while cwd[idx] != '/':
    idx -= 1
sys.path.append(cwd[:idx] + '/modules/dual_modal_yolact')

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import numpy as np
import random

import torch
import torch.backends.cudnn as cudnn

from yolact import Yolact
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess
from data import cfg, set_cfg
from data import MEANS1, MEANS2, STD1, STD2

Items = ['pedestrian', 'cyclist', 'car', 'bus', 'truck', 'traffic_light', 'traffic_sign']
Confs = [0.20, 0.20, 0.60, 0.40, 0.40, 0.20, 0.20]
Topks = [10, 10, 10, 10, 10, 10, 10]

def create_random_color():
    # 功能：产生随机RGB颜色
    # 输出：color <class 'tuple'> 颜色
    
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    color = (r, g, b)
    return color

def draw_mask(img, mask, color):
    # 功能：绘制掩膜
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      mask <class 'torch.Tensor'> torch.Size([frame_height, frame_width]) 掩膜
    #      color <class 'tuple'> 颜色
    # 输出：img_numpy <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    img_gpu = torch.from_numpy(img).cuda().float()
    img_gpu = img_gpu / 255.0
    
    # 改变mask的维度 <class 'torch.Tensor'> torch.Size([frame_height, frame_width, 1])
    mask = mask[:, :, None]
    
    # color_tensor <class 'torch.Tensor'> torch.Size([3])
    color_tensor = torch.Tensor(color).to(img_gpu.device.index).float() / 255.
    
    # alpha为透明度，置1则不透明
    alpha = 0.45
    
    mask_color = mask.repeat(1, 1, 3) * color_tensor * alpha
    inv_alph_mask = mask * (- alpha) + 1
    img_gpu = img_gpu * inv_alph_mask + mask_color
    
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    return img_numpy

def draw_segmentation_result(img, mask, classname, score, box, color):
    # 功能：绘制检测结果
    # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    #      mask <class 'torch.Tensor'> torch.Size([frame_height, frame_width]) 掩膜
    #      classname <class 'str'> 类别名称
    #      score <class 'float'> 置信度
    #      box <class 'numpy.ndarray'> (4,) 矩形框坐标
    #      color <class 'tuple'> 颜色
    # 输出：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_thickness, line_thickness = 1, 2
    
    # 绘制矩形框
    x1, y1, x2, y2 = box[:]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
    
    # 绘制掩膜
    img = draw_mask(img, mask, color)
    
    # 选取矩形框左上角顶点uv坐标
    u, v = x1, y1
    text_str = '%s: %.2f' % (classname, score)
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    
    # 绘制文字底框
    # 图像，左下角uv坐标，右上角uv坐标，颜色，宽度（-1为填充）
    cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
    
    # 绘制文字
    # 图像，文字内容，左下角uv坐标，字体，大小，颜色，字体宽度
    cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img

class YolactDetector():
    def __init__(self, model='/weights/yolact_resnet50_178_100000.pth'):
        # 功能：初始化YolactDetector对象
        # 输入：model <class 'str'> 权重文件的路径
        
        # CUDA加速模式
        cuda_mode = True
        if cuda_mode:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            
        # Yolact参数配置
        trained_model = cwd[:idx] + '/modules/dual_modal_yolact' + model
        pth = SavePath.from_str(trained_model)
        config = pth.model_name + '_config'
        set_cfg(config)
        
        # 加载网络模型
        print('Loading the model...')
        self.net = Yolact()
        self.net.load_weights(trained_model)
        self.net.eval()
        if cuda_mode:
            self.net = self.net.cuda()
        self.net.detect.use_fast_nms = True
        self.net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        print('  Done.\n')
        
    def run(self, img1, img2, items=Items, score_thresholds=Confs, top_ks=Topks):
        # 功能：运行Yolact网络以检测图像中的目标
        # 输入：img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
        #      items <class 'list'> 保留的目标类别
        #      score_thresholds <class 'list'> 与items中的类别对应的置信度阈值
        #      top_ks <class 'list'> 与items中的类别对应的数量阈值
        # 输出：masks <class 'torch.Tensor'> torch.Size([N, frame_height, frame_width]) N为目标数量
        #      classes <class 'numpy.ndarray'> (N,) N为目标数量
        #      scores <class 'numpy.ndarray'> (N,) N为目标数量
        #      boxes <class 'numpy.ndarray'> (N, 4) N为目标数量
        
        # 检测图像中的目标
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        assert h1 == h2 and w1 == w2, '(h1, w1) should be the same as (h2, w2).'
        
        img1 = img1.astype(np.float32)
        img1 = cv2.resize(img1, (cfg.max_size, cfg.max_size))
        img1 = (img1 - MEANS1) / STD1
        img1 = img1[:, :, ::-1] # To RGB
        
        img2 = img2.astype(np.float32)
        img2 = cv2.resize(img2, (cfg.max_size, cfg.max_size))
        img2 = (img2 - MEANS2) / STD2
        img2 = img2[:, :, ::-1] # To RGB
        
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1) # [3, h, w] in RGB
        img2 = torch.from_numpy(img2.copy()).permute(2, 0, 1) # [3, h, w] in RGB
        
        img_tensor = torch.cat([img1, img2], dim=0)
        batch = img_tensor.unsqueeze(0).cuda().float()
        preds = self.net(batch)
        
        # 建立每个目标的掩膜masks、类别classes、置信度scores、边界框boxes的一一对应关系
        with torch.no_grad():
            h, w = h1, w1
            
            with timer.env('Postprocess'):
                save = cfg.rescore_bbox
                cfg.rescore_bbox = True
                # 检测结果
                t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=min(score_thresholds))
                cfg.rescore_bbox = save
                
            with timer.env('Copy'):
                idx = t[1].argsort(0, descending=True)
                masks = t[3][idx]
                ides, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
                
        # classes中存储目标类别的名称
        classes = []
        for i in range(ides.shape[0]):
            name = cfg.dataset.class_names[ides[i]]
            classes.append(name)
        classes = np.array(classes)
        
        # remain_idxs为保留的检测结果
        remain_idxs = []
        num = len(items)
        num_items = []
        for i in range(num):
            num_items.append(0)
        
        for i in range(classes.shape[0]):
            if classes[i] in items:
                # 提取目标类别在items中的索引
                idx = items.index(classes[i])
                
                # 按top_k和score_threshold提取检测结果
                if num_items[idx] < top_ks[idx] and scores[i] > score_thresholds[idx]:
                    remain_idxs.append(i)
                    num_items[idx] += 1
                    
        masks = masks[remain_idxs]
        classes = classes[remain_idxs]
        scores = scores[remain_idxs]
        boxes = boxes[remain_idxs]
        
        return masks, classes, scores, boxes
        
if __name__ == '__main__':
    img1 = cv2.imread('/home/lishangjie/data/KITTI/kitti_dual/images/000008.png')
    img2 = cv2.imread('/home/lishangjie/data/KITTI/kitti_dual/lidar_ddm_jet/000008.png')
    
    detector = YolactDetector()
    masks, classes, scores, boxes = detector.run(img1, img2)
    
    print('masks', type(masks), masks.shape)
    print('classes', type(classes), classes.shape)
    print('scores', type(scores), scores.shape)
    print('boxes', type(boxes), boxes.shape)
    
    for i in range(masks.shape[0]):
        mask = masks[i]
        classname = str(classes[i])
        score = float(scores[i])
        box = boxes[i]
        color = create_random_color()
        img1 = draw_segmentation_result(img1, mask, classname, score, box, color)
        img2 = draw_segmentation_result(img2, mask, classname, score, box, color)
        
    cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    
    # 显示图像时按Esc键终止程序
    if cv2.waitKey(0) == 27:
        cv2.destroyWindow("img1")
        cv2.destroyWindow("img2")

