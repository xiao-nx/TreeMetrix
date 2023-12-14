#coding:utf8
import math
import random

import numpy as np
import cv2

import cmlswarp

def mlswarp(image, label, x_num_grid=4, y_num_grid=4, is_drawn=False): 
  height = image.shape[0]
  width = image.shape[1]

  step_h = math.floor(height/y_num_grid)
  step_w = math.floor(width/x_num_grid)

  offset_h = math.floor((height - (step_h * y_num_grid))/2)
  offset_w = math.floor((width - (step_w * x_num_grid))/2)

  src_points = []
  dst_points = []

  for j in range(1, y_num_grid):
    for i in range(1, x_num_grid):
      rd_x = random.randint(-step_w//4, step_w//4)
      rd_y = random.randint(-step_h//4, step_h//4)
      x = int(i * step_w + rd_x + offset_w)
      y = int(j * step_h + rd_y + offset_h)
      src_points.append([i * step_w, j * step_h])
      dst_points.append([i * step_w + rd_x, j * step_h + rd_y])
    
  src_points = np.array(src_points, dtype=np.int32).reshape(-1, 2)
  dst_points = np.array(dst_points, dtype=np.int32).reshape(-1, 2)

  # mlswarp warp函数说明
  # warp_image = mlswarp.warp(image, src_points, dst_points)
  #
  # 必须参数：
  #   image: uint8图像
  #   src_points: int32 N*2原始坐标ndarray矩阵，每一行是一组x,y坐标
  #   dst_points: 同src_points
  #
  # 可选参数
  #   method: 字符串 Similarity Rigid Piecewise （默认Similarity）
  #   alpha: double （默认1.0）
  #   grid_size: int （默认5）
  warp_image = cmlswarp.warp(image, src_points, dst_points,method="Rigid", alpha=1.0, grid_size=5)
  if label is None:
    warp_label = None
  else:
    warp_label = cmlswarp.warp(label, src_points, dst_points,method="Rigid", alpha=1.0, grid_size=5)

  if is_drawn:
    for src_point, dst_point in zip(src_points, dst_points):
      src_point = (src_point[0], src_point[1])
      dst_point = (dst_point[0], dst_point[1])
      #cv2.line(warp_image, src_point, dst_point, (0, 0, 255), 2)
      #cv2.circle(warp_image, src_point, 4, (0, 0, 255), 2)
      #cv2.circle(warp_image, dst_point, 4, (0, 255, 0), 2)
                        
  return warp_image, warp_label


if __name__ == "__main__":
  image = cv2.imread('./img001.jpg')
  warp_image, _ = mlswarp(image, None, is_drawn=True)
  concat_image = np.concatenate((image, warp_image), axis=1)
  cv2.imwrite('warp_result.jpg', concat_image)



    
