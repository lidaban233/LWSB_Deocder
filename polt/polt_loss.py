import matplotlib.pyplot as plt
import numpy as np

# 读取训练日志文件
with open('train_log_20241130110205.txt', 'r') as f:
    lines = f.readlines()

loss = []
# 提取每一行的loss值
for line in lines:
    line = line.split(':')[-1].strip()
    loss.append(float(line))

# 生成每个epoch的编号
epochs = [i for i in range(1, len(loss) + 1)]

# 对损失进行平滑处理，使用卷积平滑
smooth_loss = np.convolve(loss, np.ones(10)/10, mode='valid')

# 绘制损失曲线
plt.figure(figsize=(10, 6))

# 绘制原始的训练损失曲线（实线）
plt.plot(epochs, loss, color='tab:blue', linestyle='-', linewidth=2, label='Training Loss')

# 绘制平滑后的训练损失曲线（虚线）
plt.plot(epochs[:len(smooth_loss)], smooth_loss, color='tab:red', linestyle='-', linewidth=2, label='Smoothed Training Loss', alpha=0.8)

# 设置标题和标签
plt.title('Training Loss vs. Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)

# 调整图例
plt.legend(loc='upper right', fontsize=12)

# 设置网格显示，增强图表可读性
plt.grid(True, linestyle='--', alpha=0.7)

# 自动调整布局，避免标签重叠
plt.tight_layout()

# 保存图像为SVG格式
plt.savefig('./fig/training_loss_vs_epochs_with_smoothed.svg', format='svg')

# 显示图表
plt.show()
