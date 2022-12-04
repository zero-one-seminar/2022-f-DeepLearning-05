# 4.21 二乗和誤差
import numpy as np

# 二乗和誤差の実装
def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 二乗和誤差が小さいほど教師データにより適合している
# 正解ラベル(教師データ)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# NNWの出力が"[2]の確率が最も高い"だったとき
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(sum_squared_error(np.array(y), np.array(t)))

# NNWの出力が"[7]の確率が最も高い"だったとき
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squared_error(np.array(y), np.array(t)))