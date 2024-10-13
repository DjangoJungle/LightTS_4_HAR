# HAR Classification using BackboneTimeMixer

本项目使用 TimesNet 模型在 UCI-HAR 数据集上进行人体活动识别（HAR）任务。

## 目录结构

```plaintext
project/
├── data/
│   └── UCI_HAR_Dataset/        # 放置下载的 UCI-HAR 数据集
├── models/
│   ├── backbone.py             # TimesNet 模型代码
│   └── layers.py               # 模型所需的层代码
├── utils/
│   └── data_preprocessing.py   # 数据预处理代码
├── train.py                    # 训练模型的脚本
├── evaluate.py                 # 评估模型的脚本
├── requirements.txt            # 所需的 Python 包列表
└── README.md                   # 项目说明
```

python train.py --device cuda --num_epochs 50

python evaluate.py --device cuda
