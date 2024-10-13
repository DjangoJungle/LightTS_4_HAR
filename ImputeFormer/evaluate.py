import torch
from models.backbone import HARModel
from utils.data_preprocessing import load_data, preprocess_data, get_data_loaders
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 拼接所有批次的预测结果和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 打印分类报告
    print('Classification Report:')
    print(classification_report(all_labels, all_preds))

    # 绘制混淆矩阵
    plot_confusion_matrix(cm)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate HARModel on UCI-HAR Dataset')
    parser.add_argument('--data_dir', type=str, default='data/UCI_HAR_Dataset', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='impute_former.pth', help='Path to saved model')
    parser.add_argument('--n_steps', type=int, default=128, help='Number of time steps')
    parser.add_argument('--n_features', type=int, default=9, help='Number of features')
    parser.add_argument('--n_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for evaluation')

    args = parser.parse_args()

    # 加载和预处理数据
    X, y = load_data(args.data_dir)
    X, y = preprocess_data(X, y)
    _, val_loader = get_data_loaders(X, y, batch_size=args.batch_size)

    # 初始化模型并加载权重
    model = HARModel(
        n_steps=args.n_steps,
        n_features=args.n_features,
        n_classes=args.n_classes,
        d_model=64,
        n_heads=8,
        n_layers=2,
        dim_proj=16,
        node_embedding_dim=16,
        feed_forward_dim=128,
        dropout=0.1,
    )

    device = torch.device(args.device)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 评估模型
    evaluate_model(model, val_loader, device=device)
