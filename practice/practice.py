import numpy as np

def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 教師データ
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# ニューラルネットワークの出力
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
y3 = [
    [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
    [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
]

# 二乗和誤差をy1, y2のそれぞれで出力し、結果を比べよう。



# クロスエントロピー誤差をy1, y2のそれぞれで出力し、結果を比べよう。



# バッチ処理に対応したクロスエントロピー誤差を実装しよう。
"""
データが1つの場合と、データがバッチとしてまとめられて入力される場合の両方に対応するように実装してください。
データが1つの場合の配列の形状に注意しましょう。y1とy3の形状を見比べてみると良いです。
バッチサイズはyから取得できます。
"""
def batch_cross_entropy_error(y, t):
    return

# y1, y2, y3に対して、バッチ処理に対応したクロスエントロピー誤差を実行してみよう。