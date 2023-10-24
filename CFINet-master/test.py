# import os.path as osp
# import  os
# import shutil
# # shutil.move('E://dataset/test.json','./tools/img_split/Annotations')
# # shutil.move('E://dataset/val.json','./tools/img_split/Annotations')
# # shutil.move('E://dataset/train.json','./tools/img_split/Annotations')
# os.makedirs('./work_dirs/roi_feats/cfinet')
# import os

# # 指定文件夹的路径
# folder_path = 'E://dataset/Images'
#
# # 使用os.listdir()列出文件夹下的所有文件和子文件夹
# files_and_folders = os.listdir(folder_path)
#
# # 筛选出所有的文件，而不包括子文件夹
# # files = [f for f in files_and_folders if os.path.isfile(os.path.join(folder_path, f))]
# #
# # # 现在，"files" 列表中包含了文件夹下所有的文件的文件名
# # for file in files:
# #     print(file)
# for file in files_and_folders:
#     path=osp.join(folder_path,file)
#     if osp.isfile(path):
#         shutil.move(path,'./tools/img_split/Images')


import heapq
#
# # 初始为空的大根堆
# H = []
#
# # 要插入的关键字列表
# keywords = [5, 13, 24, 35, 22, 79, 86, 75, 64, 53, 29]
#
# import heapq
# ll=[1,4,2,3,5]
# print(ll,'原始数组')
# heapq.heapify(ll)
# print(ll,'小根堆')
# # 此时若想得到大顶堆
# newl = [(-i, ll[i]) for i in range(len(ll))]
# print(newl,'插入负数后的小根堆')
# heapq.heapify(newl) #以插入的负数做小根堆，越大的数字插入的负数就越小，所以这样就相当于做了大根堆
# # 此时已经按照第一个值变成了小顶堆，即变成了逆序
# max_heap = list()
# while newl:
#     _, s = heapq.heappop(newl) #删除并返回 newl中的最小元素
#     max_heap.append(s)
# print(max_heap,'输出的大根堆'
# tensor([0.4022, 0.4650, 0.4837, 0.4721, 0.5291], device='cuda:0')
from mmdet.datasets import build_dataset
import  cv2
import os
import torch
#
# import json
# with open('./tools/CFINet1/eval_20231008_171421.json','r') as file:
#     data=json.load(file)
# print(data['metric'])



# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default='./01016.jpg', help='Image file')
    parser.add_argument('--config',default='./tools/CFINet/faster_rcnn_r50_fpn_cfinet_1x.py', help='Config file')
    parser.add_argument('--checkpoint',default='./tools/CFINet/latest.pth', help='Checkpoint file')
    parser.add_argument('--out-file', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='voc',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):

    model = init_detector(args.config, args.checkpoint, device=args.device)
    file=cv2.imread(args.img)

    h,w,_=file.shape
    i=j=0
    list1=[]
    # while i<=w-800  and j<=h-800:
    for j in range(0,h,800):
        for i in range(0,w,800):
            if(j+800<h) and (i+800<w):
                list1.append(file[j:j+800,i:i+800])
            elif(j+800>=h)and(i+800>=w):
                list1.append(file[h-800:h-1,w-800:w-1])
            else:
                if j+800>=h:
                    list1.append(file[h-800:h-1,i:i+800])
                else:
                    list1.append(file[j:j + 800,w-800:w-1])
    list2=[]
    for i,img in enumerate(list1):
        list2.append('./output/input{}.jpg'.format(i))
        cv2.imwrite('./output/input{}.jpg'.format(i),img)
    for i,img in enumerate(list2):
        result=inference_detector(model,img)
        show_result_pyplot(
            model,
            img,
            result,
            title='resulty_{}'.format(i),
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=args.out_file)
    # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)
    # # test a single image
    # result = inference_detector(model, args.img)
    # # show the results
    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     palette=args.palette,
    #     score_thr=args.score_thr,
    #     out_file=args.out_file)



    # test a single image
    # result = inference_detector(model, args.img)
    # # show the results
    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     palette=args.palette,
    #     score_thr=args.score_thr,
    #     out_file=args.out_file)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
