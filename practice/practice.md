---
marp: true
math: katex
header: "損失関数"
footer: '2022/12/7 ゼロイチゼミ <a href="https://twitter.com/nu_zero_one" style="color:white">@nu_zero_one</a>'
theme: 01semi
paginate: true
---

<!--
_class: title
_paginate: false
-->

# 損失関数

### ゆってぃー

---

## 二乗和誤差

- y1, y2それぞれで二乗和誤差を出力し、結果を考察しよう

```python
import numpy as np

def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 教師データ
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# ニューラルネットワークの出力
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
```

---

## 交差エントロピー誤差
</br>
</br>
</br>

$$
E=-\dfrac{1}{N}\sum _{k}t_{k}\log y_{k}
$$

---
## 交差エントロピー誤差

- y1, y2で交差エントロピー誤差を出力し、結果を考察しよう

```python
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 教師データ
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# ニューラルネットワークの出力
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
```

---

## [バッチ対応版] 交差エントロピー誤差

- バッチ処理に対応した交差エントロピー誤差を実装しよう
    - 以下の数式を参考にしてください
    - データが1つの場合と、データがバッチとしてまとめられて入力される場合の両方に対応するように実装してください

$$
E=-\dfrac{1}{N}\sum _{n}\sum _{k}t_{n,k}\log y_{n,k}
$$



