# dual_modal_perception

ROS package for dual modal perception (rgbt)

## 安装
 - 使用Anaconda设置环境依赖，并确保系统已经安装ROS
   ```
   # 创建名为yolact-env的虚拟环境
   conda create -n yolact-env python=3.6.9
   conda activate yolact-env
   
   # 安装PyTorch、torchvision以及其他功能包
   conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
   pip install cython
   pip install opencv-python pillow pycocotools matplotlib
   pip install -U scikit-learn
   
   # 使Python3与ROS兼容
   pip install catkin_tools
   pip install rospkg
   ```
 - 建立工作空间并拷贝这个库
   ```Shell
   mkdir -p ros_ws/src
   cd ros_ws/src
   git clone https://github.com/shangjie-li/dual_modal_perception.git --recursive
   cd ..
   catkin_make
   ```
 - 准备双模态检测的模型文件，并保存至目录`dual_modal_perception/modules/dual_modal_yolact/weights`
 - 准备单模态检测的模型文件，并保存至目录`dual_modal_perception/modules/yolact/weights`

## 参数配置
 - 编写相机标定参数`fusion_perception/conf/calibration_image.yaml`
   ```
   %YAML:1.0
   ---
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [859, 0, 339, 0, 0, 864, 212, 0, 0, 0, 1, 0]
   Height: 2.0
   DepressionAngle: 0.0
   ```
 - 在上述参数文件中
   ```
   ProjectionMat:
     该3x4矩阵为通过相机内参标定得到的projection_matrix。
   Height:
     相机高度，单位米。
   DepressionAngle:
     相机俯角，单位度。
   ```
 - 修改算法相关参数`dual_modal_perception/conf/param.yaml`
   ```
   print_time:                         True
   print_objects_info:                 False
   record_time:                        True
   record_objects_info:                True
  
   sub_image1_topic:                   /image1 # RGB image
   sub_image2_topic:                   /image2 # Thermal image
   calibration_image_file:             calibration_image.yaml
  
   display_image_raw:                  False
   display_image_segmented:            False
  
   display_2d_modeling:                False
   display_gate:                       False
  
   display_3d_modeling:                True
   display_frame:                      True
   display_obj_state:                  True
  
   processing_mode: DT # D - detection, DT - detection and tracking
   modality: RGB # RGB - RGB image, T - Thermal image, RGBT - Both
  
   blind_update_limit:                 0
   frame_rate:                         10
   max_id:                             10000
   ```
    - `sub_image1_topic`指明订阅的可见光图像话题。
    - `sub_image2_topic`指明订阅的红外图像话题。

## 运行
 - 加载参数文件至ROS参数服务器
   ```
   cd fusion_perception/conf
   rosparam load param.yaml
   ```
 - 启动算法`dual_modal_perception`
   ```
   cd dual_modal_perception/scripts
   python3 demo.py
   ```

