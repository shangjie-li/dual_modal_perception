import os
import sys
cwd = os.getcwd().rstrip('scripts')
sys.path.append(os.path.join(cwd, 'modules/yolov5-test'))

import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import argparse
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from utils.general import check_img_size, \
    box_iou, non_max_suppression, scale_coords, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import torch_distributed_zero_first, InfiniteDataLoader
from utils.datasets import Dataset, img_formats, help_url
from utils.datasets import Image, exif_size, segments2boxes, get_hash, letterbox, xywhn2xyxy, xyxy2xywh
from functions import simplified_nms


def create_dataloader(path, imgsz, batch_size, stride, pad=0.5, rect=True,
                      rank=-1, world_size=1, workers=8, prefix=''):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      rect=rect,  # rectangular training
                                      stride=int(stride),
                                      pad=pad,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


def convert_img_path(img_path):
    sa = os.sep + 'images' + os.sep
    sb1 = os.sep + 'visible' + os.sep
    sb2 = os.sep + 'lwir' + os.sep
    return img_path.replace(sa, sb1, 1), img_path.replace(sa, sb2, 1)


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa = os.sep + 'images' + os.sep
    sb = os.sep + 'annotations_yolo' + os.sep + 'annotations' + os.sep
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


def load_dual_images(self, index):
    # loads 2 images from dataset, returns img1, img2, original hw, resized hw
    path = self.img_files[index]
    path1, path2 = convert_img_path(path)
    img1 = cv2.imread(path1)  # BGR
    img2 = cv2.imread(path2)  # BGR
    assert img1 is not None, 'Image Not Found ' + path1
    assert img2 is not None, 'Image Not Found ' + path2
    h0, w0 = img1.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img1 = cv2.resize(img1, (int(w0 * r), int(h0 * r)), interpolation=interp)
        img2 = cv2.resize(img2, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img1, img2, (h0, w0), img1.shape[:2]  # img, hw_original, hw_resized


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, rect=True, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.rect = rect
        self.stride = stride
        self.path = path

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_file():
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [parent + x for x in t]  # local to global path
                else:
                    raise Exception(f'{prefix}{p} is not a file')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache = self.cache_labels(prefix)  # cache

        # Read cache
        cache.pop('results')
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

    def cache_labels(self, prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im_file1, im_file2 = convert_img_path(im_file)
                im = Image.open(im_file1)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size (width, height)
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        # Load image
        img1, img2, (h0, w0), (h, w) = load_dual_images(self, index)

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img1, ratio, pad = letterbox(img1, shape, auto=False, scaleup=False)
        img2, ratio, pad = letterbox(img2, shape, auto=False, scaleup=False)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img1.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img1.shape[1]  # normalized width 0-1

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img1 = img1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img2 = img2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        return torch.from_numpy(img1), torch.from_numpy(img2), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img1, img2, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img1, 0), torch.stack(img2, 0), torch.cat(label, 0), path, shapes


def test(weights1, weights2, data_path, nc, batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6,
         plots=True, half_precision=True):
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    weights1 = os.path.join(cwd, 'modules/yolov5-test', weights1)
    weights2 = os.path.join(cwd, 'modules/yolov5-test', weights2)
    model1 = attempt_load(weights1, map_location=device)  # load FP32 model
    model2 = attempt_load(weights2, map_location=device)  # load FP32 model
    gs = max(int(model1.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model1.half()
        model2.half()

    # Configure
    model1.eval()
    model2.eval()
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if device.type != 'cpu':
        model1(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model1.parameters())))  # run once
        model2(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model2.parameters())))  # run once
    task = opt.task  # path to train/val/test images
    dataloader = create_dataloader(data_path, imgsz, batch_size, gs, pad=0.5, rect=True,
                                   prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model1.names if hasattr(model1, 'names') else model1.module.names)}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img1, img2, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img1 = img1.to(device, non_blocking=True)
        img1 = img1.half() if half else img1.float()  # uint8 to fp16/32
        img1 /= 255.0  # 0 - 255 to 0.0 - 1.0

        img2 = img2.to(device, non_blocking=True)
        img2 = img2.half() if half else img2.float()  # uint8 to fp16/32
        img2 /= 255.0  # 0 - 255 to 0.0 - 1.0

        targets = targets.to(device)
        nb, _, height, width = img1.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out1, _ = model1(img1, augment=False)  # inference and training outputs
            out2, _ = model2(img2, augment=False)  # inference and training outputs
            t0 += time_synchronized() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = []  # for autolabelling
            t = time_synchronized()
            out1 = non_max_suppression(out1, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            out2 = non_max_suppression(out2, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

            # Fuse
            out = []
            for i in range(nb):
                pp = torch.cat([out1[i], out2[i]], 0)
                boxes = pp[:, 0:4].cpu().numpy()
                scores = pp[:, 4].cpu().numpy().tolist()
                indices = simplified_nms(boxes, scores)
                out.append(pp[indices])

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img1[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img1[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img1, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img1, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if nc > 1 and nc < 50 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Return results
    print(f"Results saved to {save_dir}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights1', type=str, default='weights/seumm_visible/yolov5s_100ep_pretrained.pt', help='model.pt path')
    parser.add_argument('--weights2', type=str, default='weights/seumm_lwir/yolov5s_100ep_pretrained.pt', help='model.pt path')
    parser.add_argument('--data_path', type=str, default='/home/lishangjie/data/SEUMM/seumm_dual_15200/annotations_yolo/sets/val.txt', help='data path')
    parser.add_argument('--nc', type=int, default=7, help='number of classes')

    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()
    print(opt)

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.weights1, opt.weights2, opt.data_path, opt.nc,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             )
    else:
        raise NotImplementedError
