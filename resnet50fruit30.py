# -*- coding = utf-8 -*-
# @Time : 2023/6/8 20:54
# @Author : Happiness
# @File : resnet50fruit30.py
# @Software : PyCharm



####   model setting

model={'type': 'ImageClassifier',

#主干网络负责提取特征
  'backbone': {'type': 'ResNet',
               'depth': 50,
               'num_stages': 4,
               'out_indices': (3,),
               'style': 'pytorch'},

#
  'neck': {'type': 'GlobalAveragePooling'},

#细化具体的任务，做分类就用分类头，检测可能要用检测头
  'head': {'type': 'LinearClsHead',
           'num_classes': 30,
           'in_channels': 2048,  ####这个地方要与前面卷积层的输入相关
           'loss': {'type': 'CrossEntropyLoss', 'loss_weight': 1.0},
           'topk': (1,5)},

#加载预训练模型权重
  "init_cfg":{"type":"Pretrained",
              "checkpoint":"D:/Data/torch-model/hub/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth"

  }
       }



#####   dataset   setting

dataset_type='CustomDataset' # 数据集类型为自定义数据集类型CustomDataset

data_preprocessor={'num_classes': 30,#和模型的num_classes要匹配
 'mean': [123.675, 116.28, 103.53],
 'std': [58.395, 57.12, 57.375],
 'to_rgb': True}


train_pipeline=[{'type': 'LoadImageFromFile'},
 {'type': 'RandomResizedCrop', 'scale': 224},
 {'type': 'RandomFlip', 'prob': 0.5, 'direction': 'horizontal'},
 {'type': 'PackInputs'}]

test_pipeline=[{'type': 'LoadImageFromFile'},
 {'type': 'ResizeEdge', 'scale': 256, 'edge': 'short'},
 {'type': 'CenterCrop', 'crop_size': 224},
 {'type': 'PackInputs'}]


## 最关键的字段配置
train_dataloader={'batch_size': 32,
 'num_workers': 5,#pytorch用来加载样本处理样本的进程数

#训练数据集的配置（最重要）
 'dataset': {'type': 'CustomDataset',

#文件路径相关信息
  'data_root': '../../data/fruit30_split/train/',

#定义了数据集的数据处理流程
  'pipeline': [{'type': 'LoadImageFromFile'},
   {'type': 'RandomResizedCrop', 'scale': 224},
   {'type': 'RandomFlip', 'prob': 0.5, 'direction': 'horizontal'},
   {'type': 'PackInputs'}]},

#采样器的配置
 'sampler': {'type': 'DefaultSampler', 'shuffle': True}}

val_dataloader={'batch_size': 32,
 'num_workers': 5,
 'dataset': {'type': 'CustomDataset',

#文件路径相关信息
  'data_root': '../../data/fruit30_split/val/',

  'pipeline': [{'type': 'LoadImageFromFile'},
   {'type': 'ResizeEdge', 'scale': 256, 'edge': 'short'},
   {'type': 'CenterCrop', 'crop_size': 224},
   {'type': 'PackInputs'}]},
 'sampler': {'type': 'DefaultSampler', 'shuffle': False}}


val_evaluator={'type': 'Accuracy', 'topk': (1,5)}


#这里不对验证集和测试集做区分
# test_dataloader=val_dataloader
# test_evaluator=val_evaluator

###

test_dataloader={'batch_size': 32,
 'num_workers': 5,
 'dataset': {'type': 'CustomDataset',

#文件路径相关信息
  'data_root': '../../data/fruit30_split/val/',

  'pipeline': [{'type': 'LoadImageFromFile'},
   {'type': 'ResizeEdge', 'scale': 256, 'edge': 'short'},
   {'type': 'CenterCrop', 'crop_size': 224},
   {'type': 'PackInputs'}]},
 'sampler': {'type': 'DefaultSampler', 'shuffle': False}}

test_evaluator={'type': 'Accuracy', 'topk': (1,5)}

###




######  optimizer

optim_wrapper={'optimizer': {'type': 'SGD',
  'lr': 0.01,
  'momentum': 0.9,
  'weight_decay': 0.0001}}

## 优化器参数

#此处是多部参数规划器，在第30，60，90轮降低学习率，降低到原来的0.1
param_scheduler={'type': 'MultiStepLR',
 'by_epoch': True,
 'milestones': [30, 60, 90],
 'gamma': 0.1}

##训练，验证，测试的流程设置
train_cfg={'by_epoch': True, 'max_epochs': 20, 'val_interval': 1}  ####复杂的任务多些轮次
# 空字典代表使用默认参数
val_cfg={}
test_cfg={}

# #和学习率有关，batch_size越小学习率越小，也就是batch_size=256时，上面的学习率'lr': 0.1
# auto_scale_lr={'base_batch_size': 256}#256是使用了分布式的8卡训练，8*32=256，所以前面模型和数据阶段的batch_size是32




######  运行参数配置

default_scope='mmpretrain'

default_hooks={'timer': {'type': 'IterTimerHook'},
 'logger': {'type': 'LoggerHook', 'interval': 100},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'checkpoint': {'type': 'CheckpointHook', 'interval': 1, "max_keep_ckpts":2,"save_best":"auto"},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'visualization': {'type': 'VisualizationHook', 'enable': False}}

env_cfg={'cudnn_benchmark': False,
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0},
 'dist_cfg': {'backend': 'nccl'}}

vis_backends= [{'type': 'LocalVisBackend'}]

visualizer={'type': 'UniversalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}

log_level='INFO'

load_from=None

resume=False

#和随机数相关，随机种子
randomness={'seed': None, 'deterministic': False}



