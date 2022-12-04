# 4.2.3 ミニバッチ学習

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

# 訓練データ、教師データのサイズを表示する
print(x_train.shape)
print(t_train.shape)

# 訓練データからランダムに10枚取り出す
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

