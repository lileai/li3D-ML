# SuperGlue 推理计算单应性矩阵工具

## Introduction
该代码的功能用于标定两个固定相机的单应性矩阵，以及单应性矩阵的推理映射

目录：
```
.
├── .idea/
├── grpc_detection/
├── intrinsics/
├── json_file/
├── models/
├── video/
├── .gitignore
├── homography_calculate.py
├── homography_inference.py
├── LICENSE
├── README.md
└── requirements.txt

```
功能：
1. `homography_calculate.py` : 标定脚本，用于单应性矩阵的标定和可视化展示
2. `homography_calculate.py` : 推理脚本，用于单应性矩阵的实时映射和分区控制

- **grpc_detection**: 包含与gRPC相关的检测代码和配置。
- **intrinsics**: 存放相机内参。
- **json_file**: 存放H矩阵的json文件。
- **models**: 模型文件。
- **video**: 存放视频文件。

## Dependencies
以下是最小的依赖版本，无版本的依赖项在requirements.txt中
* Python 3 >= 3.5
* PyTorch >= 1.1
* OpenCV >= 3.4 (4.1.2.30 recommended for best GUI keyboard interaction, see this [note](#additional-notes))
* Matplotlib >= 3.1
* NumPy >= 1.18

Simply run the following command: `pip3 install numpy opencv-python torch matplotlib`

## 标定脚本 (homography_calculate.py)
   * 读取源数据的视频/视频流/图片路径，以及目标数据的同步的视频/视频流/图片路径
   * 在源数据中调用grpc的检测模型得到检测框
   * 利用SuperGlue模型计算匹配的关键点
   * 利用stable_planar_from_queue计算单应性矩阵
   * 利用滑动窗口稳定稳健的特征点对，优化H矩阵
   * 利用make_vis_dual_mode根据不同的模式进行可视化展示
   * 保存H矩阵和相关参数到json文件中

### 运行

```
sh ./homography_calculate.py
```
### 命令行参数:

- `-h, --help`：显示帮助信息并退出。
- `--mode {calib,show}`：选择操作模式。
- `calib`: 三拼[rgb1|overlay|rgb2]并在线计算单应性矩阵H。
- `show`: 双拼[rgb1|rgb2]直接加载H。
- `--log_level {DEBUG,INFO,WARNING,ERROR}`：设置日志级别，默认为INFO。
- `--input1 `：第一个输入视频的路径。
- `--input2 `：第二个输入视频的路径。
- `--intrinsic1_dir `：第一个相机的内参文件路径。
- `--intrinsic2_dir `：第二个相机的内参文件路径。
- `--output_dir `：输出目录。
- `--output_video `：输出视频文件的路径。
- `--best_h_json `：单应性矩阵文件路径或文件夹路径。
- `--skip `：跳过帧数，默认为5。
- `--max_length `：最大处理帧数，默认为1000000。
- `--resize `：缩放尺寸，默认为[-1]。
- `--superglue {indoor,outdoor}`：SuperGlue权重类型，默认为outdoor。
- `--max_keypoints `：最大关键点数，默认为-1。
- `--keypoint_threshold `：关键点阈值，默认为0.005。
- `--nms_radius `：非极大值抑制半径，默认为1。
- `--sinkhorn_iterations `：Sinkhorn迭代次数，默认为20。
- `--match_threshold `：匹配阈值，默认为0.2。
- `--ransac_th `：RANSAC阈值，默认为5.0。
- `--no_display`：不显示视频窗口。
- `--force_cpu`：强制使用CPU。
- `--win_len `：动态标定窗口大小，默认为100。
- `--vis_h `：可视化高度，默认为720。
- `--net NET`：是否使用网络，默认为True。
- `--boundaries `：自定义区域划分，默认为None。
- `--waitkey `：按键等待时间，标定时为0，展示时为1。

### 操作方法
1. 运行程序
2. 按任意键到下一帧
3. 如果觉得当前的映射已经比较好了，按esc退出程序

## 推理脚本 (homography_inference.py)
   * 读取源数据
   * 读取json路径下的单应性矩阵文件
   * 在源数据中调用grpc的检测模型得到检测框
   * 根据检测框在源数据图片中的位置判断当前目标所在的区域
   * 根据目标当前所在的区域和当前源数据的区域进行单应性映射，将源数据中的检测框映射到目标数据中
   * 输出检测框在源数据中的区域和对应映射后的检测框

### 运行

```
sh./homography_inference.py
```
运行结果会持续输出dict

```
{
'preset': reg_id + 1,  # 云台预置点（区域标号）
'mapped_box': dst      # 映射后的框
}
```

