# 4.2.2　交差エントロピー誤差
import numpy as np

# 交差エントロピー誤差の実装
def cross_entropy_error(y, t):
    delta = 1e-7 # log(0)が発生しないように小さい値を足す
    return -np.sum(t * np.log(y + delta))

# 正解ラベル(教師データ)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# NNWの出力が"[2]の確率が最も高い"だったとき
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# NNWの出力が"[7]の確率が最も高い"だったとき
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))