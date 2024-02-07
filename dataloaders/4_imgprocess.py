# -*- coding=utf-8 -*-
# Copyright (c) 2021 -<424236969@qq.com>
"""
Author:Towers
Create Data:2021/12/2
@Software:PyCharm
"""
import cv2 as cv
import os
import numpy as np


def match(filename, filename_list):
    for name in filename_list:
        if filename in name:
            return True

    return False


# 加载成对的数据
def dataprocess(img_path, mask_path):
    img_files = os.listdir(img_path)
    mask_files = os.listdir(mask_path)
    # 考虑到文件名没有一一对应，只有做匹配查找

    # 将抠出来的图放在out文件夹下
    out_path = "5_WP/test-JPEGImages/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for img_file_name in img_files:
        # 拆分文件名
        file_name, stuff = os.path.splitext(img_file_name)
        # 查看有没有对应的mask文件
        if match(file_name, mask_files):  # 有mask
            mask_file_name = file_name + ".jpg"

            # 获取完整的文件名路径
            img_file_full_path = os.path.join(img_path, img_file_name)
            mask_file_full_path = os.path.join(mask_path, mask_file_name)

            img_src = cv.imread(img_file_full_path)
            src_1 = np.ones_like(img_src) * 255
            mask_src = cv.imread(mask_file_full_path, 0)
            ret, mask_src = cv.threshold(mask_src, 127, 255, cv.THRESH_BINARY)

            _, mak_2 = cv.threshold(mask_src, 127, 255, cv.THRESH_BINARY_INV)

            img1_fore = cv.bitwise_and(img_src, img_src, mask=mask_src)
            img1_bg = cv.bitwise_and(src_1, src_1, mask=mak_2)
            result = img1_fore + img1_bg

            out_file_path = out_path + file_name + ".jpg"
            cv.imwrite(out_file_path, result)


if __name__ == '__main__':
    img_path = "3_deblurring"
    mask_path = "4_result_mask"
    dataprocess(img_path, mask_path)
