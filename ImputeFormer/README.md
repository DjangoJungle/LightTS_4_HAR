# HAR Classification using ImputeFormer

本项目使用 ImputeFormer 模型在 UCI-HAR 数据集上进行人体活动识别（HAR）任务。

## 目录结构

```plaintext
ImputeFormer/
├── data/
│   └── UCI_HAR_Dataset/        # 存放下载的 UCI-HAR 数据集
├── models/
│   ├── transformer/
│   │   └── embedding.py        # 定义 Transformer 模型中的嵌入层
│   ├── attention.py            # 实现自注意力机制的代码，用于 Transformer 模型
│   ├── backbone.py             # ImputeFormer 模型的骨干网络架构
│   └── mlp.py                  # 实现多层感知机（MLP）层，作为模型的组成部分
├── utils/
│   └── data_preprocessing.py   # 数据预处理代码，处理输入数据以适应模型需求
├── train.py                    # 训练模型的脚本，设置训练参数并运行训练
├── evaluate.py                 # 评估模型的脚本，加载训练好的模型并评估性能
├── requirements.txt            # 项目依赖的 Python 包列表
└── README.md                   # 项目说明文件，介绍项目用途及使用方法

```

python train.py --device cuda --num_epochs 50

python evaluate.py --device cuda
