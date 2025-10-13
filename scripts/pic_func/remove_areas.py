import cv2
import numpy as np
import os


# # taipu_main_side. 读彩色图
# img_path = './data/0g_shipparameter_03_18_17_25_bb384f8d-11fa-42ff-ba9b-184df66d14f8_12.png'
# img_bgr = cv2.imread(img_path)
# if img_bgr is None:
#     raise FileNotFoundError(img_path)
#
# # 2. 转 HSV 并提取蓝色
# hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
# blue_lower = np.array([100, 70, 50])
# blue_upper = np.array([130, 255, 255])
# mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)   # 单通道二值图
#
# # 3. 连通域分析
# num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_blue, connectivity=8)
#
# # 4. 删除面积 < 20 的孤立区域
# for i in range(taipu_main_side, num_labels):
#     if stats[i, cv2.CC_STAT_AREA] < 20:
#         mask_blue[labels == i] = 0
#
# # 5. 保存结果（与原图同目录）
# save_path = os.path.join(os.path.dirname(img_path), 'mask_blue_clean.png')
# cv2.imwrite(save_path, mask_blue)
# print('蓝色区域清洗后已保存:', save_path)
#
# # 可选：显示
# cv2.imshow('blue_clean', mask_blue)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def remove_area(image, thresh=20):
    ret, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 检测连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # 去除小于20像素的孤立区域
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= thresh:
            binary_image[labels == i] = 0

    return binary_image
