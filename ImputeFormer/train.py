import torch
import torch.nn as nn
import torch.optim as optim
from models.backbone import HARModel
from utils.data_preprocessing import load_data, preprocess_data, get_data_loaders
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=25,
    device='cpu',
):
    best_acc = 0.0
    best_model_wts = model.state_dict()  # 初始化最佳模型权重

    # 初始化用于绘图的列表
    train_acc_history = []
    val_acc_history = []
    train_f1_history = []
    val_f1_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        # 每个 epoch 包含训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                data_loader = train_loader
            else:
                model.eval()   # 设置模型为评估模式
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            # 遍历数据
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 收集所有的标签和预测，用于计算 F1 得分
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

            # 记录指标
            if phase == 'train':
                train_acc_history.append(epoch_acc.item())
                train_f1_history.append(epoch_f1)
            else:
                val_acc_history.append(epoch_acc.item())
                val_f1_history.append(epoch_f1)

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 绘制准确率曲线
    plt.figure(figsize=(10,5))
    plt.title("Accuracy over Epochs")
    plt.plot(range(1, num_epochs+1), train_acc_history, label="Train Accuracy")
    plt.plot(range(1, num_epochs+1), val_acc_history, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_over_epochs.png")
    plt.show()

    # 绘制 F1 得分曲线
    plt.figure(figsize=(10,5))
    plt.title("F1 Score over Epochs")
    plt.plot(range(1, num_epochs+1), train_f1_history, label="Train F1 Score")
    plt.plot(range(1, num_epochs+1), val_f1_history, label="Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig("f1_score_over_epochs.png")
    plt.show()

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train HARModel on UCI-HAR Dataset')
    parser.add_argument('--data_dir', type=str, default='data/UCI_HAR_Dataset', help='Path to dataset')
    parser.add_argument('--n_steps', type=int, default=128, help='Number of time steps')
    parser.add_argument('--n_features', type=int, default=9, help='Number of features')
    parser.add_argument('--n_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')

    args = parser.parse_args()

    # 加载和预处理数据
    X, y = load_data(args.data_dir)
    X, y = preprocess_data(X, y)
    train_loader, val_loader = get_data_loaders(X, y, batch_size=args.batch_size)

    # 初始化模型、损失函数和优化器
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

    # 计算并打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=args.num_epochs,
        device=device,
    )

    # 保存模型
    torch.save(model.state_dict(), 'impute_former.pth')
