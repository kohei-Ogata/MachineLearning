import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
s = os.path.join('https://archive.ics.uci.edu','ml',
                 'machine-learning-databases', 'iris','iris.data')
df = pd.read_csv(s, header=None, encoding = 'utf-8')
#1-100行目の目的関数の抽出
y = df.iloc[0:100, 4].values
#iris-setosaを-1, Iris-versicolorを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
#1-100行目の1,3列目の抽出
X = df.iloc[0:100, [0,2]].values
#品種setosaのプロット（赤の○）
plt.scatter(X[:50,0],X[:50,1],color='red', marker='o',label='setosa')
#品種versicolorのプロット（青の×）
plt.scatter(X[50:100,0],X[50:100,1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length(cm)')
plt.ylabel('petal length(cm)')
plt.legend(loc = 'upper left')
plt.show()