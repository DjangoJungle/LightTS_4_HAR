# HAR Classification using BackboneTimeMixer

本项目使用 TimeMixer 模型在 UCI-HAR 数据集上进行人体活动识别（HAR）任务。

## 目录结构

```plaintext
project/
├── data/
│   └── UCI_HAR_Dataset/        # 用于存储下载的 UCI-HAR 数据集，该数据集用于人体活动识别（HAR）任务
├── models/
│   ├── transformer/
│   │   └── embedding.py        # 定义 TimeMixer 模型中 Transformer 部分的嵌入层
│   ├── autoformer.py           # 实现 TimeMixer 模型的序列分解模块
│   ├── backbone.py             # 定义 TimeMixer 模型的骨干网络
│   ├── layers.py               # 包含 TimeMixer 模型中所需的特定层的定义
│   └── revin.py                # 实现模型的层归一化和反归一化功能，便于处理输入数据
├── utils/
│   └── data_preprocessing.py   # 处理和预处理原始数据，以便用于模型训练和评估
├── train.py                    # 训练脚本，用于设置参数并启动模型训练
├── evaluate.py                 # 评估脚本，用于加载训练好的模型并在测试数据上评估性能
├── requirements.txt            # 列出项目所需的 Python 包及其版本
└── README.md                   # 项目说明文件，介绍项目目的、使用方法及结构

```

python train.py --device cuda --num_epochs 50

python evaluate.py --device cuda
