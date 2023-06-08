# -*- coding = utf-8 -*-
# @Time : 2023/6/8 21:59
# @Author : Happiness
# @File : tuili.py
# @Software : PyCharm


from mmpretrain import ImageClassificationInferencer

inferencer=ImageClassificationInferencer("./resnet50fruit30.py",pretrained="exp/best_accuracy_top1_epoch_19.pth",device='cuda')#指定配置文件和预训练模型，并且使用显卡加速

image_list=["./fruit/1.JPG","./fruit/2.JPG","./fruit/3.JPG","./fruit/4.JPG","./fruit/5.JPG","./fruit/6.JPG","./fruit/7.JPG","./fruit/8.JPG","./fruit/9.JPG",
            "./fruit/10.JPG","./fruit/11.JPG","./fruit/12.JPG","./fruit/13.JPG","./fruit/14.JPG","./fruit/15.JPG","./fruit/16.JPG","./fruit/17.JPG",
            "./fruit/18.JPG","./fruit/19.JPG","./fruit/20.JPG","./fruit/21.JPG","./fruit/22.JPG","./fruit/23.JPG","./fruit/24.JPG"]

results=inferencer(image_list)

print("")  ###空一行让圣女果和其他顶头对齐

#mmpretrain自带的工具和接口无法在图片上显示中文标签，所以直接预测标签结果并打印
for i in range(24):
    print(results[i]['pred_class'])
