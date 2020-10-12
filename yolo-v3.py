# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import xml.etree.ElementTree as Et
import time
import cv2
import random
import bisect
import math
import matplotlib.pyplot as plt
import csv
from visdom import Visdom


class DataSet(data.Dataset):  # 使用VOC2012数据集
    def __init__(self, is_train=True, data_transform=None):
        self.data_transform = data_transform
        self.img_dir = './VOC2012/JPEGImages/'
        self.data_set_info_dir = './VOC2012/ImageSets/Main/'
        self.obj_info_dir = './VOC2012/Annotations/'
        self.re_size = 416  # resize to 416 * 416
        self.cls_num = 20  # class number
        self.data_set_info = self.collect_data_set_info()  # 得到训练集、验证集图片编号
        self.cls = []  # 数据集类别
        self.train_set, self.val_set = self.train_val()  # 输出训练集、验证集
        if args.re_cal_anchor:  # 是否重新计算anchor
            self.anchor = self.cal_anchor(train_set)
        else:  # 否则使用此前计算的anchor
            self.anchor = np.array([[350.1096, 285.9541], [211.8508, 258.0122], [323.6481, 143.7598],
                                    [113.8253, 225.3156], [164.6536, 137.5494], [70.7844, 133.5198],
                                    [120.5636, 71.9585], [45.9448, 63.8777], [20.6699, 28.4295]])
        self.data_set = self.train_set if is_train else self.val_set

    def collect_data_set_info(self):
        info_dic = []  # 列表用于存储字典
        for file in os.listdir(self.data_set_info_dir):
            if len(file) >= 8 and file[-8:] == '_val.txt':  # 通过文件名后缀和长度判断
                txt_data = open(self.data_set_info_dir + file)
                txt_lines = txt_data.readlines()  # 按行分割txt
                txt_data.close()
                for single_line in txt_lines:
                    single_line_split = single_line.split()  # 分割每行的内容
                    if single_line_split[1] == '1':
                        info_dic.append({'set': 'val', 'pic_name': single_line_split[0]})
            elif len(file) >= 10 and file[-10:] == '_train.txt':  # 通过文件名后缀和长度判断
                txt_data = open(self.data_set_info_dir + file)
                txt_lines = txt_data.readlines()  # 按行分割txt
                txt_data.close()
                for single_line in txt_lines:
                    single_line_split = single_line.split()  # 分割每行的内容
                    if single_line_split[1] == '1':
                        info_dic.append({'set': 'train', 'pic_name': single_line_split[0]})
        return info_dic

    def train_val(self):
        train_set, val_set = [], []
        for _i in self.data_set_info:
            cls, bndbox = self.xml_dic(self.obj_info_dir + _i['pic_name'] + '.xml')  # 读取对应xml获取类别和位置
            if _i['set'] == 'train':
                train_set.append({'img_path': self.img_dir + _i['pic_name'] + '.jpg', 'cls': cls, 'bndbox': bndbox})
            else:
                val_set.append({'img_path': self.img_dir + _i['pic_name'] + '.jpg', 'cls': cls, 'bndbox': bndbox})
        return train_set, val_set

    def xml_dic(self, file):  # 解析xml文件得到类别和位置信息
        one_file_cls = []
        one_file_bndbox = []
        root_node = Et.parse(file).getroot()
        c1_nodes = list(root_node)
        for _i in c1_nodes:
            if _i.tag == 'object':
                c2_nodes = list(_i)
                for _j in c2_nodes:
                    if _j.tag == 'name':
                        obj_name = _j.text
                        if obj_name not in self.cls:
                            self.cls.append(obj_name)
                        obj_cls = self.cls.index(obj_name)
                    if _j.tag == 'bndbox':
                        c3_nodes = list(_j)
                        for _k in c3_nodes:
                            if _k.tag == 'xmin':
                                x_min = int(_k.text)
                            elif _k.tag == 'ymin':
                                y_min = int(_k.text)
                            elif _k.tag == 'xmax':
                                x_max = int(_k.text)
                            elif _k.tag == 'ymax':
                                y_max = int(_k.text)
                        bndbox = np.array([x_min, y_min, x_max, y_max])
                one_file_cls.append(obj_cls), one_file_bndbox.append(bndbox)
        return one_file_cls, one_file_bndbox

    def __getitem__(self, index):
        sample = self.data_set[index]
        img, label = self.resize_and_labeled(sample)  # resize并计算标签值
        if self.data_transform is not None:
            img = self.data_transform(img)
        return img, label

    def resize_and_labeled(self, sample):  # resize并计算标签值
        img = cv2.imread(sample['img_path'])
        h, w, _ = img.shape
        ratio = self.re_size / max(h, w)  # 得到缩放比
        new_img = cv2.resize(np.asarray(img), (round(ratio * w), round(ratio * h)))  # 等比例缩放
        new_h, new_w, _ = new_img.shape
        # 使用保持x y比例方式resize，当宽高不相等时，进行填充
        top, bottom, left, right = 0, 0, 0, 0
        if new_w == new_h:  # 对宽高相等的情况，图片填充为零
            pass
        elif new_w == self.re_size:  # 若缩放后w与缩放尺寸相同，则填充h
            need_pad = self.re_size - new_h
            top, bottom = int(need_pad / 2), math.ceil(need_pad / 2)
        else:  # 若缩放后h与缩放尺寸相同，则填充w
            need_pad = self.re_size - new_w
            left, right = int(need_pad / 2), math.ceil(need_pad / 2)
        img_border = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # 计算标签值
        label_13 = np.zeros((13, 13, 3, (5 + self.cls_num)))  # grid 13 * 13 标签值
        label_26 = np.zeros((26, 26, 3, (5 + self.cls_num)))  # grid 26 * 26 标签值
        label_52 = np.zeros((52, 52, 3, (5 + self.cls_num)))  # grid 52 * 52 标签值

        for i in range(len(sample['bndbox'])):
            resize_bndbox = sample['bndbox'][i].reshape(2, 2) * ratio + np.array([left, top])  # 对框坐标缩放和平移
            x_min, y_min, x_max, y_max = resize_bndbox.flatten()  # 压平
            x_c, y_c, w, h = (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min  # 中心点、宽、高

            max_iou = 0
            max_iou_idx = 0
            for j in range(9):
                overlap_area = min(w, self.anchor[j][0]) * min(h, self.anchor[j][1])  # 重叠面积
                iou = overlap_area / (w * h + self.anchor[j][0] * self.anchor[j][1] - overlap_area)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = j

            if max_iou_idx < 3:  # 13 * 13 grid
                cell_pixel = 32  # 32 = 416 / 13
                scale_x_c, scale_y_c = x_c / cell_pixel, y_c / cell_pixel  # 以方格大小尺度化
                c_x, c_y = int(scale_x_c), int(scale_y_c)
                label_13[c_x, c_y, max_iou_idx, 0:2] = scale_x_c - c_x, scale_y_c - c_y
                label_13[c_x, c_y, max_iou_idx, 2:4] = np.log(w / self.anchor[max_iou_idx][0]), \
                                                       np.log(h / self.anchor[max_iou_idx][1])
                label_13[c_x, c_y, max_iou_idx, 4] = 1
                label_13[c_x, c_y, max_iou_idx, 5 + sample['cls'][i]] = 1

            elif max_iou_idx < 6:  # 26 * 26 grid
                cell_pixel = 16  # 16 = 416 / 26
                scale_x_c, scale_y_c = x_c / cell_pixel, y_c / cell_pixel  # 以方格大小尺度化
                c_x, c_y = int(scale_x_c), int(scale_y_c)
                label_26[c_x, c_y, max_iou_idx - 3, 0:2] = scale_x_c - c_x, scale_y_c - c_y
                label_26[c_x, c_y, max_iou_idx - 3, 2:4] = np.log(w / self.anchor[max_iou_idx][0]), \
                                                           np.log(h / self.anchor[max_iou_idx][1])
                label_26[c_x, c_y, max_iou_idx - 3, 4] = 1
                label_26[c_x, c_y, max_iou_idx - 3, 5 + sample['cls'][i]] = 1

            else:  # 52 * 52 grid
                cell_pixel = 8  # 8 = 416 / 52
                scale_x_c, scale_y_c = x_c / cell_pixel, y_c / cell_pixel  # 以方格大小的尺度化
                c_x, c_y = int(scale_x_c), int(scale_y_c)
                label_52[c_x, c_y, max_iou_idx - 6, 0:2] = scale_x_c - c_x, scale_y_c - c_y
                label_52[c_x, c_y, max_iou_idx - 6, 2:4] = np.log(w / self.anchor[max_iou_idx][0]), \
                                                           np.log(h / self.anchor[max_iou_idx][1])
                label_52[c_x, c_y, max_iou_idx - 6, 4] = 1
                label_52[c_x, c_y, max_iou_idx - 6, 5 + sample['cls'][i]] = 1

        label_13 = label_13.reshape((13 * 13, 3, (5 + self.cls_num)))
        label_26 = label_26.reshape((26 * 26, 3, (5 + self.cls_num)))
        label_52 = label_52.reshape((52 * 52, 3, (5 + self.cls_num)))

        # 聚合, 形状[13 * 13 + 26 * 26 + 52 * 52, 3, 5 + self.cls_num]
        label = np.concatenate((label_13, label_26, label_52), axis=0)
        return img_border, label

    def __len__(self):
        return len(self.data_set)

    def cal_anchor(self, train_set):
        w_h_list = []
        for sample in train_set:
            img = cv2.imread(sample['img_path'])
            h, w, _ = img.shape
            ratio = self.re_size / max(h, w)  # 得到缩放比
            for _i in sample['bndbox']:
                x_min, y_min, x_max, y_max = _i * ratio
                w, h = x_max - x_min, y_max - y_min
                w_h_list.append([w, h])
        w_h_x9 = KMeansAnchor(w_h_list, 9).repeat()
        anchor_area_sort = np.argsort(-w_h_x9[:, 0] * w_h_x9[:, 1])
        w_h_x9_sort = w_h_x9[anchor_area_sort]
        return w_h_x9_sort


class KMeansAnchor:
    def __init__(self, w_h, k):  # k类别
        self.w_h = np.array(w_h)
        self.k = k

    def get_first_center(self):  # 获取首个类别中心点
        return self.w_h[random.choice(range(len(self.w_h)))]

    def get_centers(self):  # 获取所有类别中心点（首个 + 剩余）
        centers = [self.get_first_center()]
        remainder_k = self.k - 1
        for _i in range(remainder_k):
            centers.append(self.w_h[self.select_center_by_weight(centers)])  # 基于权重选择中心点
        return np.array(centers)

    def select_center_by_weight(self, centers):  # 基于权重选择中心点
        weight_sum = 0
        weight_sum_list = []
        for _i in self.w_h:
            weight_sum += np.min(self.one_sub_iou(centers, _i))  # 计算1-iou
            weight_sum_list.append(weight_sum)
        return bisect.bisect_left(weight_sum_list, random.uniform(0, weight_sum_list[-1])) - 1

    @staticmethod
    def one_sub_iou(centers, sample):  # 计算1-iou
        result = []
        w_s, h_s = sample
        for center in centers:
            w_c, h_c = center
            overlap_area = min(w_c, w_s) * min(h_c, h_s)  # 重叠面积
            iou = overlap_area / (w_c * h_c + w_s * h_s - overlap_area)
            result.append(1 - iou)
        return np.array(result)

    def repeat(self):  # 重复更新中心直到稳定
        repeat_time = 0
        latest_centers = self.get_centers()  # 最新的聚类中心
        while True:
            new_centers = self.refresh_centers(latest_centers)  # 不断更新的聚类中心
            repeat_time += 1
            if (new_centers == latest_centers).all():
                break
            latest_centers = new_centers
        return new_centers

    def refresh_centers(self, centers):
        cluster_index = []
        new_centers = []
        for _j in range(self.k):
            cluster_index.append([])
        for _i in range(len(self.w_h)):
            min_index = np.argmin(self.one_sub_iou(centers, self.w_h[_i]))
            cluster_index[min_index].append(_i)
        for _c in range(self.k):
            new_centers.append(np.mean(self.w_h[cluster_index[_c]], axis=0))
        return np.array(new_centers)


class ConvBnLeakyrelu(nn.Module):
    def __init__(self, in_chnls, out_chnls, ksize=3, stride=1, padding=1):
        super(ConvBnLeakyrelu, self).__init__()
        self.conv = nn.Conv2d(in_chnls, out_chnls, ksize, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_chnls)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, chnls):
        super(ResidualBlock, self).__init__()
        self.layer_1 = ConvBnLeakyrelu(chnls, chnls // 2, ksize=1, padding=0)
        self.layer_2 = ConvBnLeakyrelu(chnls // 2, chnls)

    def forward(self, x):
        residual = x
        x = self.layer_1(x)
        x = self.layer_2(x)
        x += residual
        return x


class YoloV3(nn.Module):
    def __init__(self, cls_num=20):
        super(YoloV3, self).__init__()
        self.cls_num = cls_num

        self.layer_1 = ConvBnLeakyrelu(3, 32)  # 416 * 416
        self.layer_2 = ConvBnLeakyrelu(32, 64, stride=2)  # 208 *208
        self.layer_3_4 = self.blocks(64, '1', 1)  # 208 *208
        self.layer_5 = ConvBnLeakyrelu(64, 128, stride=2)  # 104 *104
        self.layer_6_9 = self.blocks(128, '2', 2)  # 104 *104
        self.layer_10 = ConvBnLeakyrelu(128, 256, stride=2)  # 52 * 52
        self.layer_11_26 = self.blocks(256, '3', 8)  # 52 * 52
        self.layer_27 = ConvBnLeakyrelu(256, 512, stride=2)  # 26 * 26
        self.layer_28_43 = self.blocks(512, '4', 8)  # 26 * 26
        self.layer_44 = ConvBnLeakyrelu(512, 1024, stride=2)  # 13 * 13
        self.layer_45_52 = self.blocks(1024, '5', 4)  # 13 * 13

        # large object predict
        self.layer_53_57 = self.convbnleakyrelu_x5(1024, 512)
        self.layer_58 = ConvBnLeakyrelu(512, 1024, ksize=3)  # 13 * 13
        self.layer_59 = nn.Conv2d(1024, 3 * (5 + cls_num), 1, stride=1)  # 13 * 13

        # mid object predict
        self.layer_60 = ConvBnLeakyrelu(512, 256, ksize=1, padding=0)  # 13 * 13
        self.layer_61 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 26 * 26
        self.layer_62_66 = self.convbnleakyrelu_x5(768, 256)  # 26 * 26
        self.layer_67 = ConvBnLeakyrelu(256, 512, ksize=3)  # 26 * 26
        self.layer_68 = nn.Conv2d(512, 3 * (5 + cls_num), 1, stride=1)  # 26 * 26

        # small object predict
        self.layer_69 = ConvBnLeakyrelu(256, 128, ksize=1, padding=0)  # 26 * 26
        self.layer_70 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 52 * 52
        self.layer_71_75 = self.convbnleakyrelu_x5(384, 128)  # 52 * 52
        self.layer_76 = ConvBnLeakyrelu(128, 256, ksize=3)  # 52 * 52
        self.layer_77 = nn.Conv2d(256, 3 * (5 + cls_num), 1, stride=1)  # 52 * 52

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                m.bias.data.fill_(0)  # 偏差初始为零
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer_1(x)  # 416 * 416 - 32
        x = self.layer_2(x)  # 208 * 208 - 64
        x = self.layer_3_4(x)
        x = self.layer_5(x)  # 104 * 104 - 128
        x = self.layer_6_9(x)
        x = self.layer_10(x)  # 52 * 52 - 256
        short_cut_small = self.layer_11_26(x)
        x = self.layer_27(short_cut_small)  # 26 * 26 - 512
        short_cut_mid = self.layer_28_43(x)
        x = self.layer_44(short_cut_mid)  # 13 * 13 - 1024
        x = self.layer_45_52(x)

        # large object predict
        up_sample_mid = self.layer_53_57(x)  # 13 * 13 - 512
        x = self.layer_58(up_sample_mid)  # 13 * 13 - 1024
        y1 = self.layer_59(x)  # 13 * 13
        # mid object predict
        x = self.layer_60(up_sample_mid)  # 13 * 13 - 256
        x = self.layer_61(x)  # 26 * 26 - 256
        x = torch.cat((short_cut_mid, x), 1)  # 26 * 26 - 768
        up_sample_small = self.layer_62_66(x)  # 26 * 26 - 256
        x = self.layer_67(up_sample_small)  # 26 * 26 - 512
        y2 = self.layer_68(x)  # 26 * 26

        # small object predict
        x = self.layer_69(up_sample_small)  # 26 * 26 - 256
        x = self.layer_70(x)  # 52 * 52 - 256
        x = torch.cat((short_cut_small, x), 1)  # 52 * 52 - 384
        x = self.layer_71_75(x)  # 52 * 52 - 128
        x = self.layer_76(x)  # 52 * 52 - 256
        y3 = self.layer_77(x)  # 52 * 52

        return y1, y2, y3

    @staticmethod
    def blocks(in_chnls, block_num, multi_num):
        blocks = nn.Sequential()
        for i in range(multi_num):
            blocks.add_module('blocks_' + block_num + "-x" + str(multi_num), ResidualBlock(in_chnls))
        return blocks

    @staticmethod
    def convbnleakyrelu_x5(in_chnls, out_chnls):
        return nn.Sequential(
            ConvBnLeakyrelu(in_chnls, out_chnls, ksize=1, padding=0),
            ConvBnLeakyrelu(out_chnls, out_chnls * 2, ksize=3),
            ConvBnLeakyrelu(out_chnls * 2, out_chnls, ksize=1, padding=0),
            ConvBnLeakyrelu(out_chnls, out_chnls * 2, ksize=3),
            ConvBnLeakyrelu(out_chnls * 2, out_chnls, ksize=1, padding=0)
        )


class Multiwork:
    def __init__(self, model_path=None):
        self.model_path = model_path

        if not os.path.exists(args.save_directory):  # 新建模型保存文件夹
            os.makedirs(args.save_directory)

        # 图片左上角位置矩阵
        cx_cy = torch.zeros((3549, 3, 2))
        for i in range(3549):
            if i < 169:
                j = i
                cx_cy[i, :] = torch.tensor([j // 13, j % 13])
            elif i < 845:
                j = i - 169
                cx_cy[i, :] = torch.tensor([j // 26, j % 26])
            else:
                j = i - 845
                cx_cy[i, :] = torch.tensor([j // 52, j % 52])
        self.cx_cy = cx_cy.unsqueeze(dim=0)

        # 每个cell的像素尺寸矩阵
        cell_pixel = torch.zeros((3549, 3, 2))
        cell_pixel[:169, :] = torch.tensor([32, 32])
        cell_pixel[169:845, :] = torch.tensor([16, 16])
        cell_pixel[845:, :] = torch.tensor([8, 8])
        self.cell_pixel = cell_pixel.unsqueeze(dim=0)

        # anchor宽、高矩阵
        anchor = torch.from_numpy(DataSet().anchor)  # 数据集anchor
        anchor_w_h = torch.zeros((3549, 3, 2))
        anchor_w_h[:169, :], anchor_w_h[169:845, :], anchor_w_h[845:, :] = anchor[0:3], anchor[3:6], anchor[6:9]
        self.anchor_w_h = anchor_w_h.unsqueeze(dim=0)

        self.start_time = time.time()  # 开始时间，用于打印计算用时
        data_transform = transforms.Compose([transforms.ToTensor()])
        self.train_set = DataSet(data_transform=data_transform)
        self.val_set = DataSet(is_train=False, data_transform=data_transform)
        self.re_size = self.train_set.re_size
        print('cls = ', self.train_set.cls)
        self.cls_color = [(0, 255, 0), (255, 153, 18), (255, 97, 3), (65, 105, 225),
                          (255, 0, 0), (128, 42, 42), (160, 32, 240), (176, 48, 96),
                          (11, 23, 70), (0, 0, 255), (255, 255, 0), (252, 230, 201),
                          (220, 220, 220), (25, 25, 112), (255, 0, 255), (218, 112, 214),
                          (176, 23, 31), (210, 180, 140), (0, 255, 127), (112, 128, 105)]  # 类别对应颜色，供后续绘图使用

        work = args.work
        if work not in ['train', 'test', 'finetune', 'predict']:
            print("args.work should be one of ['train', 'test', 'finetune', 'predict']")
        elif work == 'train':
            self.train()
        elif self.model_path is None:
            print('Please input model_path')
        elif work == 'finetune':
            self.finetune()
        elif args.work == "test":  # 测试
            loss, coord, objectness, class_pred = self.test(self.load_model_path, is_path=True)
            print(f'loss: {mean_loss}  coord: {max_loss}  objectness: {min_loss}  class_pred: {class_pred}')
            collect_loss = (['loss', 'coord', 'objectness', 'class_pred'], [loss, coord, objectness, class_pred])  # 采集loss
            save_loss_path = self.load_model_path[:-3] + '_test_result.csv'
            self.writelist2csv(collect_loss, save_loss_path)
            print(f'--Save complete!\n--save_loss_path: {save_loss_path}\n')
            print('Test complete!')
        elif work == 'predict':
            self.predict_img()

    def train(self):
        print(f"Start Train!  data_set_len: {self.train_set.__len__()}")
        cx_cy = torch.cat([self.cx_cy for _ in range(args.batch_size)], dim=0).to(device)
        cell_pixel = torch.cat([self.cell_pixel for _ in range(args.batch_size)], dim=0).to(device)
        anchor_w_h = torch.cat([self.anchor_w_h for _ in range(args.batch_size)], dim=0).to(device)
        load_data = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        model = YoloV3().to(device).train()

        criterion_mse = nn.MSELoss(reduction='sum')
        criterion_bce = nn.BCEWithLogitsLoss(reduction='sum')
        current_lr = args.lr
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)
        epoch_count, loss_record, cost_time_record = [], [], []
        collect_loss = [['train_loss', 'train_loss_coord', 'train_loss_objectness', 'train_loss_class_pred',
                         'test_loss', 'test_loss_coord', 'test_loss_objectness', 'test_loss_class_pred']]
        for e in range(args.epochs):
            epoch_loss_coord, epoch_loss_objectness, epoch_loss_class_pred, epoch_loss = [], [], [], []
            for index, (img, label) in enumerate(load_data):
                # print(img.size())
                img = img.to(device)
                label = torch.from_numpy(label.numpy().astype('float32')).to(
                    device)  # [b, 13*13+26*26+52*52, 3, cls_num]
                optimizer.zero_grad()
                output1, output2, output3 = model(img)

                output1 = output1.permute(0, 2, 3, 1).view(-1, 169, 75)
                output2 = output2.permute(0, 2, 3, 1).view(-1, 676, 75)
                output3 = output3.permute(0, 2, 3, 1).view(-1, 2704, 75)
                output = torch.cat((output1, output2, output3), dim=1).view(-1, 3549, 3, 25)

                obj_index = label[:, :, :, 4] == 1  # 索引(有物体)
                noobj_index = label[:, :, :, 4] == 0  # 索引(无物体)
                # print(index, obj_index)
                # print(index, noobj_index)

                gt_x_y = (label[obj_index][:, 0:2] + cx_cy[obj_index]) * cell_pixel[obj_index]
                gt_w_h = torch.exp(label[obj_index][:, 2:4]) * anchor_w_h[obj_index]
                x_y = (torch.sigmoid(output[obj_index][:, 0:2]) + cx_cy[obj_index]) * cell_pixel[obj_index]
                w_h = torch.exp(output[obj_index][:, 2:4]) * anchor_w_h[obj_index]
                is_overlap, iou = self.cal_iou(gt_x_y, gt_w_h, x_y, w_h)
                label[obj_index][:, 4][is_overlap] = iou

                loss_coord = args.weight_coord * (
                        criterion_mse(label[obj_index][:, 0:2], torch.sigmoid(output[obj_index][:, 0:2])) +
                        criterion_mse(label[obj_index][:, 2:4], output[obj_index][:, 2:4]))
                loss_objectness = criterion_mse(label[obj_index][:, 4], torch.sigmoid(output[obj_index][:, 4])) + \
                                  args.weight_noobj * criterion_mse(label[noobj_index][:, 4],
                                                                    torch.sigmoid(output[noobj_index][:, 4]))
                loss_class_pred = criterion_bce(output[obj_index][:, 5:], label[obj_index][:, 5:])
                loss = loss_coord + loss_objectness + loss_class_pred
                # print(f'index: {index}  label[obj_index][:, 0:2]: {label[obj_index][:, 0:2]}'
                #       f'torch.sigmoid(output[obj_index][:, 0:2]): {torch.sigmoid(output[obj_index][:, 0:2])}'
                #       f'label[obj_index][:, 2:4]: {label[obj_index][:, 2:4]}'
                #       f'output[obj_index][:, 2:4]: {output[obj_index][:, 2:4]}'
                #       f'label[obj_index][:, 4]: {label[obj_index][:, 4]}'
                #       f'torch.sigmoid(output[obj_index][:, 4]): {torch.sigmoid(output[obj_index][:, 4])}'
                #       f'label[noobj_index][:, 4]: {label[noobj_index][:, 4]}'
                #       f'torch.sigmoid(output[noobj_index][:, 4]): {torch.sigmoid(output[noobj_index][:, 4])}'
                #       f'output[obj_index][:, 5:], label[obj_index][:, 5:])')

                epoch_loss_coord.append(loss_coord.item())
                epoch_loss_objectness.append(loss_objectness.item())
                epoch_loss_class_pred.append(loss_class_pred.item())
                epoch_loss.append(loss.item())
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()
                # print(loss.item(), loss_coord.item(), loss_objectness.item(), loss_class_pred.item())

            m_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            m_epoch_loss_coord = sum(epoch_loss_coord) / len(epoch_loss_coord)
            m_epoch_loss_objectness = sum(epoch_loss_objectness) / len(epoch_loss_objectness)
            m_epoch_loss_class_pred = sum(epoch_loss_class_pred) / len(epoch_loss_class_pred)
            test_loss, test_coord, test_objectness, test_cls_pred = self.test(model.state_dict())
            # 供visdom显示
            epoch_count.append(e + 1)
            loss_record.append([m_epoch_loss, m_epoch_loss_coord, m_epoch_loss_objectness, m_epoch_loss_class_pred,
                                test_loss, test_coord, test_objectness, test_cls_pred])
            cost_time_record.append(time.time() - self.start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart2', opts=opts2)
        collect_loss.extend(loss_record)
        save_model_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M', time.localtime(self.start_time))
                                       + '_train_epoch_' + str(e) + ".pt")
        save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M', time.localtime(self.start_time))
                                      + '_train_loss.csv')
        torch.save(model.state_dict(), save_model_path)
        self.writelist2csv(collect_loss, save_loss_path)
        print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def test(self, input_model, is_path=False):
        cx_cy = torch.cat([self.cx_cy for _ in range(args.batch_size)], dim=0).to(device1)
        cell_pixel = torch.cat([self.cell_pixel for _ in range(args.batch_size)], dim=0).to(device1)
        anchor_w_h = torch.cat([self.anchor_w_h for _ in range(args.batch_size)], dim=0).to(device1)
        load_data = DataLoader(self.val_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        # 输出数据集大小
        if is_path:
            print(f"Start Test!  len_dataset: {self.val_set.__len__()}")
        model = YoloV3().to(device1)
        if is_path:  # 模型参数加载
            model.load_state_dict(torch.load(input_model))
        else:
            model.load_state_dict(input_model)
        model.eval()  # 关闭参数梯度

        criterion_mse = nn.MSELoss(reduction='sum')
        criterion_bce = nn.BCEWithLogitsLoss(reduction='sum')
        current_lr = args.lr

        epoch_loss_coord, epoch_loss_objectness, epoch_loss_class_pred, epoch_loss = [], [], [], []
        for index, (img, label) in enumerate(load_data):
            img = img.to(device1)
            label = torch.from_numpy(label.numpy().astype('float32')).to(device1)
            output1, output2, output3 = model(img)

            output1 = output1.permute(0, 2, 3, 1).view(-1, 169, 75)
            output2 = output2.permute(0, 2, 3, 1).view(-1, 676, 75)
            output3 = output3.permute(0, 2, 3, 1).view(-1, 2704, 75)
            output = torch.cat((output1, output2, output3), dim=1).view(-1, 3549, 3, 25)

            obj_index = label[:, :, :, 4] == 1  # 索引(有物体)
            noobj_index = label[:, :, :, 4] == 0  # 索引(无物体)

            gt_x_y = (label[obj_index][:, 0:2] + cx_cy[obj_index]) * cell_pixel[obj_index]
            gt_w_h = torch.exp(label[obj_index][:, 2:4]) * anchor_w_h[obj_index]
            x_y = (torch.sigmoid(output[obj_index][:, 0:2]) + cx_cy[obj_index]) * cell_pixel[obj_index]
            w_h = torch.exp(output[obj_index][:, 2:4]) * anchor_w_h[obj_index]
            is_overlap, iou = self.cal_iou(gt_x_y, gt_w_h, x_y, w_h)
            label[obj_index][:, 4][is_overlap] = iou

            loss_coord = args.weight_coord * (
                    criterion_mse(label[obj_index][:, 0:2], torch.sigmoid(output[obj_index][:, 0:2])) +
                    criterion_mse(label[obj_index][:, 2:4], output[obj_index][:, 2:4]))
            loss_objectness = criterion_mse(label[obj_index][:, 4], torch.sigmoid(output[obj_index][:, 4])) + \
                              args.weight_noobj * criterion_mse(label[noobj_index][:, 4],
                                                                torch.sigmoid(output[noobj_index][:, 4]))
            loss_class_pred = criterion_bce(output[obj_index][:, 5:], label[obj_index][:, 5:])
            loss = loss_coord + loss_objectness + loss_class_pred

            epoch_loss_coord.append(loss_coord.item())
            epoch_loss_objectness.append(loss_objectness.item())
            epoch_loss_class_pred.append(loss_class_pred.item())
            epoch_loss.append(loss.item())

        m_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        m_epoch_loss_coord = sum(epoch_loss_coord) / len(epoch_loss_coord)
        m_epoch_loss_objectness = sum(epoch_loss_objectness) / len(epoch_loss_objectness)
        m_epoch_loss_class_pred = sum(epoch_loss_class_pred) / len(epoch_loss_class_pred)
        return m_epoch_loss, m_epoch_loss_coord, m_epoch_loss_objectness, m_epoch_loss_class_pred

    def predict(self):
        print(f"Start Predict!")
        model = YoloV3().to(device)
        model.load_state_dict(torch.load(self.model_path))
        # confidence_threshold = 0.3  # 物体判定阈值
        confidence_threshold = 1e-3  # 物体判定阈值
        iou_threshold = 0.5  # 重叠判定阈值
        probability_threshold = 0.5
        cx_cy = self.cx_cy.squeeze().to(device)
        cell_pixel = self.cell_pixel.squeeze().to(device)
        anchor_w_h = self.anchor_w_h.squeeze().to(device)
        # anchor_w_h = anchor_w_h.reshape(3549 * 3, 2)
        # loss_wh = args.weight_coord * criterion_mse(label[obj_index][:, 2:4],  # 框长宽loss
        #                                             anchor_w_h * torch.exp(y[obj_index][:, 2:4]))

        # 视频目标识别
        video_capture = cv2.VideoCapture('./test.mp4')  # 捕获视频
        fps = video_capture.get(cv2.CAP_PROP_FPS)  # 帧速
        fsize = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fnums = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)  # 帧数
        available, frame = video_capture.read()
        # video_writer = cv2.VideoWriter(self.save_video_filename, self.fourcc, fps, fsize)

        # 帧图片等比例变形并填充
        w, h = fsize
        ratio = self.re_size / max(h, w)  # 得到缩放比
        new_w, new_h = round(ratio * w), round(ratio * h)
        # 使用保持x y比例方式resize，当宽高不相等时，进行填充
        top, bottom, left, right = 0, 0, 0, 0
        if new_w == new_h:  # 对宽高相等的情况，图片填充为零
            pass
        elif new_w == self.re_size:  # 若缩放后w与缩放尺寸相同，则填充h
            need_pad = self.re_size - new_h
            top, bottom = int(need_pad / 2), math.ceil(need_pad / 2)
        else:  # 若缩放后h与缩放尺寸相同，则填充w
            need_pad = self.re_size - new_w
            left, right = int(need_pad / 2), math.ceil(need_pad / 2)
        frame_num = 1
        while available:
            print(frame_num)
            new_img = cv2.resize(np.asarray(frame), (new_w, new_h))  # 等比例缩放
            img_border = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img_tensor = transforms.ToTensor()(img_border).unsqueeze(dim=0).to(device)
            y1, y2, y3 = model(img_tensor)

            y1 = y1.permute(0, 2, 3, 1).view(1, 169, 75)
            y2 = y2.permute(0, 2, 3, 1).view(1, 676, 75)
            y3 = y3.permute(0, 2, 3, 1).view(1, 2704, 75)
            y = torch.cat((y1, y2, y3), dim=1).view(1, 3549, 3, 25).squeeze(0)  # shape [3549 3 25]

            confidence = torch.sigmoid(y[:, :, 4])
            # print('confidence.size()', confidence.size())
            obj_index = confidence > confidence_threshold  # 索引(有物体)
            # print(f'obj_index {obj_index}')
            objs = y[obj_index][:, :]
            objs_cx_cy = cx_cy[obj_index]
            objs_cell_pixel = cell_pixel[obj_index]
            objs_anchor_w_h = anchor_w_h[obj_index]
            if objs.size()[0] == 0:  # 在当前探测阈值下不包含物体
                continue

            frame_num += 1
            objs_confidence = objs[:, 4]
            objs_sort_confidence_index = torch.argsort(objs_confidence)
            objs_sort = objs[objs_sort_confidence_index]
            objs_cx_cy_sort = objs_cx_cy[objs_sort_confidence_index]
            objs_cell_pixel_sort = objs_cell_pixel[objs_sort_confidence_index]
            objs_anchor_w_h_sort = objs_anchor_w_h[objs_sort_confidence_index]
            # print(f'objs_sort.size {objs_sort.size()}')
            object_reserve = []
            for i in range(objs_sort.size()[0]):
                if torch.sigmoid(objs_sort[i][4]) < confidence_threshold:
                    continue
                base_x_y = (torch.sigmoid(objs_sort[i][0:2]) + objs_cx_cy_sort[i]) * objs_cell_pixel_sort[i]
                base_w_h = torch.exp(objs_sort[i][2:4]) * objs_anchor_w_h_sort[i]
                object_reserve.append({'x_y': base_x_y, 'w_h': base_w_h, 'cls_pr': objs_sort[i][5:]})
                objs_sort[i][4] = -8
                if objs_sort.size()[0] - i > 1:
                    base_x_y = base_x_y.repeat((objs_sort.size()[0] - 1 - i, 1))
                    base_w_h = base_w_h.repeat((objs_sort.size()[0] - 1 - i, 1))

                    compare_x_y = (torch.sigmoid(objs_sort[i + 1:, 0:2]) + objs_cx_cy_sort[
                                                                           i + 1:]) * objs_cell_pixel_sort[i + 1:]
                    compare_w_h = torch.exp(objs_sort[i + 1:, 2:4]) * objs_anchor_w_h_sort[i + 1:]
                    is_overlap, iou = self.cal_iou(base_x_y, base_w_h, compare_x_y, compare_w_h)
                    suppression = iou > iou_threshold
                    if objs_sort[i + 1:][is_overlap][suppression].size()[0] != 0:
                        objs_sort[i + 1:][is_overlap][suppression][:, 4] = -8

            img_show = frame.copy()
            for _obj in object_reserve:
                max_cls_idx = torch.argmax(_obj['cls_pr'])
                cls = self.data_set.cls[max_cls_idx]
                score = _obj['cls_pr'][max_cls_idx]
                x_y = _obj['x_y']
                w_h = _obj['w_h']
                left_top = (int(x_y[0] - w_h[0] / 2), int(x_y[1] - w_h[1] / 2))
                right_bottom = (int(x_y[0] + w_h[0] / 2), int(x_y[1] + w_h[1] / 2))
                cv2.rectangle(img_show, left_top, right_bottom, color=self.cls_color[max_cls_idx], thickness=1)  # 画方框
                plt.imshow(img_show)
                # print(left_top[0], left_top[1], cls, torch.sigmoid(score).item())
                plt.text(left_top[0], left_top[1], cls + ' ' + str(torch.sigmoid(score).item())[:4], fontsize=8,
                         # 标类别字符
                         verticalalignment="bottom", horizontalalignment="left",
                         bbox=dict(boxstyle='round,pad=0', fc=tuple(np.array(self.cls_color[max_cls_idx]) / 255),
                                   ec=tuple(np.array(self.cls_color[max_cls_idx]) / 255), lw=0, alpha=1, ))
            plt.show()

            # video_writer.write(frame_boxed)
            available, frame = video_capture.read()  # 获取下一帧
        # video_writer.release()
        video_capture.release()

    def predict_img(self):
        print(f"Start Predict!")
        model = YoloV3().to(device)
        model.load_state_dict(torch.load(self.model_path))
        # confidence_threshold = 0.3  # 物体判定阈值
        confidence_threshold = 0.1  # 物体判定阈值
        iou_threshold = 0.5  # 重叠判定阈值
        probability_threshold = 0.5
        cx_cy = self.cx_cy.squeeze().to(device)
        cell_pixel = self.cell_pixel.squeeze().to(device)
        anchor_w_h = self.anchor_w_h.squeeze().to(device)
        # anchor_w_h = anchor_w_h.reshape(3549 * 3, 2)
        # loss_wh = args.weight_coord * criterion_mse(label[obj_index][:, 2:4],  # 框长宽loss
        #                                             anchor_w_h * torch.exp(y[obj_index][:, 2:4]))

        load_data = DataLoader(self.val_set, batch_size=1, shuffle=True, drop_last=True)

        for index, (img, label) in enumerate(load_data):
            print(index)
            img = img.to(device)

            y1, y2, y3 = model(img)

            y1 = y1.permute(0, 2, 3, 1).view(1, 169, 75)
            y2 = y2.permute(0, 2, 3, 1).view(1, 676, 75)
            y3 = y3.permute(0, 2, 3, 1).view(1, 2704, 75)
            y = torch.cat((y1, y2, y3), dim=1).view(1, 3549, 3, 25).squeeze(0)  # shape [3549 3 25]

            confidence = torch.sigmoid(y[:, :, 4])
            # print('confidence.size()', confidence.size())
            obj_index = confidence > confidence_threshold  # 索引(有物体)
            # print(f'obj_index {obj_index}')
            objs = y[obj_index][:, :]
            objs_cx_cy = cx_cy[obj_index]
            objs_cell_pixel = cell_pixel[obj_index]
            objs_anchor_w_h = anchor_w_h[obj_index]
            if objs.size()[0] == 0:  # 在当前探测阈值下不包含物体
                continue

            objs_confidence = objs[:, 4]
            objs_sort_confidence_index = torch.argsort(objs_confidence)
            objs_sort = objs[objs_sort_confidence_index]
            objs_cx_cy_sort = objs_cx_cy[objs_sort_confidence_index]
            objs_cell_pixel_sort = objs_cell_pixel[objs_sort_confidence_index]
            objs_anchor_w_h_sort = objs_anchor_w_h[objs_sort_confidence_index]
            # print(f'objs_sort.size {objs_sort.size()}')
            object_reserve = []
            for i in range(objs_sort.size()[0]):
                if torch.sigmoid(objs_sort[i][4]) < confidence_threshold:
                    continue
                base_x_y = (torch.sigmoid(objs_sort[i][0:2]) + objs_cx_cy_sort[i]) * objs_cell_pixel_sort[i]
                base_w_h = torch.exp(objs_sort[i][2:4]) * objs_anchor_w_h_sort[i]
                object_reserve.append({'x_y': base_x_y, 'w_h': base_w_h, 'cls_pr': objs_sort[i][5:]})
                objs_sort[i][4] = -8
                if objs_sort.size()[0] - i > 1:
                    base_x_y = base_x_y.repeat((objs_sort.size()[0] - 1 - i, 1))
                    base_w_h = base_w_h.repeat((objs_sort.size()[0] - 1 - i, 1))

                    compare_x_y = (torch.sigmoid(objs_sort[i + 1:, 0:2]) + objs_cx_cy_sort[
                                                                           i + 1:]) * objs_cell_pixel_sort[i + 1:]
                    compare_w_h = torch.exp(objs_sort[i + 1:, 2:4]) * objs_anchor_w_h_sort[i + 1:]
                    is_overlap, iou = self.cal_iou(base_x_y, base_w_h, compare_x_y, compare_w_h)
                    suppression = iou > iou_threshold
                    if objs_sort[i + 1:][is_overlap][suppression].size()[0] != 0:
                        objs_sort[i + 1:][is_overlap][suppression][:, 4] = -8
            img_show = img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
            # plt.imshow(img_show)
            # plt.show()
            print(len(object_reserve))
            for _obj in object_reserve:
                max_cls_idx = torch.argmax(_obj['cls_pr'])
                cls = self.val_set.cls[max_cls_idx]
                score = _obj['cls_pr'][max_cls_idx]
                x_y = _obj['x_y']
                w_h = _obj['w_h']
                left_top = (int(x_y[0] - w_h[0] / 2), int(x_y[1] - w_h[1] / 2))
                right_bottom = (int(x_y[0] + w_h[0] / 2), int(x_y[1] + w_h[1] / 2))
                cv2.rectangle(img_show, left_top, right_bottom, color=self.cls_color[max_cls_idx], thickness=1)  # 画方框
                plt.imshow(img_show)
                # print(left_top[0], left_top[1], cls, torch.sigmoid(score).item())
                plt.text(left_top[0], left_top[1], cls + ' ' + str(torch.sigmoid(score).item())[:4], fontsize=8,
                         # 标类别字符
                         verticalalignment="bottom", horizontalalignment="left",
                         bbox=dict(boxstyle='round,pad=0', fc=tuple(np.array(self.cls_color[max_cls_idx]) / 255),
                                   ec=tuple(np.array(self.cls_color[max_cls_idx]) / 255), lw=0, alpha=1, ))
            plt.show()


        #     # video_writer.write(frame_boxed)
        #     available, frame = video_capture.read()  # 获取下一帧
        # # video_writer.release()
        # video_capture.release()

    @staticmethod
    def cal_iou(x1_y1, w1_h1, x2_y2, w2_h2):
        left = torch.max(x1_y1[:, 0] - 0.5 * w1_h1[:, 0], x2_y2[:, 0] - 0.5 * w2_h2[:, 0])
        right = torch.min(x1_y1[:, 0] + 0.5 * w1_h1[:, 0], x2_y2[:, 0] + 0.5 * w2_h2[:, 0])
        top = torch.max(x1_y1[:, 1] - 0.5 * w1_h1[:, 1], x2_y2[:, 1] - 0.5 * w2_h2[:, 1])
        bottom = torch.min(x1_y1[:, 1] + 0.5 * w1_h1[:, 1], x2_y2[:, 1] + 0.5 * w2_h2[:, 1])
        is_overlap = (right > left) & (bottom > top)
        overlap_area = (right[is_overlap] - left[is_overlap]) * (bottom[is_overlap] - top[is_overlap])
        union_area = ((w1_h1[:, 0][is_overlap] * w1_h1[:, 1][is_overlap]) +
                      (w2_h2[:, 0][is_overlap] * w2_h2[:, 1][is_overlap])) - overlap_area
        return is_overlap, overlap_area / union_area

    @staticmethod
    def writelist2csv(list_data, csv_name):  # 列表写入.csv
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for one_slice in list_data:
                csv_writer.writerow(one_slice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.000001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='save_model',
                        help='learnt models are saving here')
    parser.add_argument('--re-cal-mean-std', type=bool, default=False,
                        help='if need to re calculate dateset mean and std')
    parser.add_argument('--class-num', type=int, default=20,
                        help='class num')
    parser.add_argument('--re-cal-anchor', type=bool, default=False,
                        help='re calculate anchor')
    parser.add_argument('--weight-coord', type=float, default=5, metavar='M',
                        help='coord weight')
    parser.add_argument('--weight-noobj', type=float, default=0.5, metavar='M',
                        help='noobj weight')
    parser.add_argument('--work', type=str, default='predict',  # train, eval, finetune, predict
                        help='training, eval, predicting or finetuning')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cpu")
    # visdom可视化设置
    vis = Visdom(env="yolo-v3")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss of mean/max/min in epoch',
        "xlabel": 'epoch',
        "ylabel": 'loss',
        "width": 600,
        "height": 400,
        "legend": ['train_loss', 'train_coord', 'train_objectness', 'train_class_pred',
                   'test_loss', 'test_coord', 'test_objectness', 'test_class_pred']
    }
    opts2 = {
        "title": 'cost time with epoch',
        "xlabel": 'epoch',
        "ylabel": 'time in second',
        "width": 400,
        "height": 300,
        "legend": ['cost time']
    }

    Multiwork(model_path='save_model/202008272048_train_epoch_9.pt')
