# 4.2.4 [バッチ対応版] 交差エントロピー誤差の実装
import numpy as np

"""
y : 出力
t : 教師データ
yの次元数が1の場合、データの形状を整形する。
バッチの枚数で正規化し、1枚あたりの平均の交差エントロピー誤差を計算。
"""
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    delta = 1e-7
    return - np.sum(t * np.log(y + delta) / batch_size)
    # 教師データがone-hotベクトルでない場合は次
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta) / batch_size)

