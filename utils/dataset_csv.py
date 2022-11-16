# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: MyPoseNet
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2021-06-17
# --------------------------------------------------------
"""

import csv
import re
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging
from PIL import Image


def CenterLabelHeatMapResize(img_height, img_width, c_x, c_y, resize_h, resize_w, sigma):   # 根据关键点的坐标生成heatmap
    if max(c_x, c_y) == 0:
        return np.zeros([resize_h, resize_w])
    else:
        c_x = int(c_x * (resize_w / img_width))
        c_y = int(c_y * (resize_h / img_height))

        Y1 = np.linspace(1, resize_w, resize_w)
        X1 = np.linspace(1, resize_h, resize_h)
        [X, Y] = np.meshgrid(Y1, X1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap * 255
        return heatmap


class DatasetPoseCSV(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)

            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:
                self.name_list.append(img_sort)
                points_num = int(self.labels[img_sort][1])  # 当前图像中的人数
                img_sort = img_sort+points_num
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, scale, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis = -1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):

        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图

        img = Image.open(img_file)
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = self.preprocess(self.resize_w, self.resize_h, img0, self.scale, 1)
        searchContext = "_"

        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                points_all[k, n, :] = point
                sigma = 3
                heatmap0 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, self.scale, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  # ,
        }
