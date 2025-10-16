import numpy as np

labels = np.load("/root/WZR/TRIBE/datasets/CIFAR-100-C/labels.npy")

# 检查类别切换情况
print(labels[:300])