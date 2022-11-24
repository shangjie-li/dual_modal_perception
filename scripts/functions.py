import cv2
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image


def get_stamp(header):
    return header.stamp.secs + 0.000000001 * header.stamp.nsecs


def publish_image(pub, data, frame_id='base_link'):
    assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
    header = Header(stamp=rospy.Time.now())
    header.frame_id = frame_id
    
    msg = Image()
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = 'rgb8'
    msg.data = np.array(data).tostring()
    msg.header = header
    msg.step = msg.width * 1 * 3
    
    pub.publish(msg)


def display(img, v_writer, win_name='result'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    v_writer.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        v_writer.release()
        return False
    else:
        return True


def print_info(frame, stamp, delay, labels, scores, boxes, locs, file_name='result.txt'):
    time_str = 'frame:%d  stamp:%.3f  delay:%.3f' % (frame, stamp, delay)
    print(time_str)
    with open(file_name, 'a') as fob:
        fob.write(time_str + '\n')
    for i in range(len(labels)):
        info_str = 'box:%d %d %d %d  score:%.2f  label:%s' % (
            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], labels[i]
        )
        if locs[i] is not None:
            info_str += '  loc:(%.2f, %.2f)' % (locs[i][0], locs[i][1])
        print(info_str)
        with open(file_name, 'a') as fob:
            fob.write(info_str + '\n')
    print()
    with open(file_name, 'a') as fob:
        fob.write('\n')


def simplified_nms(boxes, scores, iou_thres=0.25):
    '''
    Args:
        boxes: (n, 4), xyxy format
        scores: list(float)
    Returns:
        indices: list(int), indices to keep
    '''
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idx = np.argsort(scores)[::-1]
    indices = []
    
    while len(idx) > 0:
        i = idx[0]
        indices.append(i)
        if len(idx) == 1:
            break
        idx = idx[1:]
        xx1 = np.clip(x1[idx], a_min=x1[i], a_max=np.inf)
        yy1 = np.clip(y1[idx], a_min=y1[i], a_max=np.inf)
        xx2 = np.clip(x2[idx], a_min=0, a_max=x2[i])
        yy2 = np.clip(y2[idx], a_min=0, a_max=y2[i])
        w, h = np.clip((xx2 - xx1), a_min=0, a_max=np.inf), np.clip((yy2 - yy1), a_min=0, a_max=np.inf)
        
        inter = w * h
        union = area[i] + area[idx] - inter
        iou = inter / union
        idx = idx[iou < iou_thres]
        
    return indices
