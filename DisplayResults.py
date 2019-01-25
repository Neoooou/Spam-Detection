import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


tst_accs = [0.9598, .9484, .9766, 0.9913]
precisions = [0.9598, .8432, .9864, 0.9879]
recalls = [0.8691, .7743, .9212, 0.9738]
f1 = [0.84, .8073, .9099, 0.9683]

idx = np.arange(len(tst_accs))
width = 0.4
fig, ax = plt.subplots()


r1 = plt.bar(idx - width, tst_accs, width/2, color='SkyBlue', label='测试集准确率')

r2 = plt.bar(idx - width/2, precisions, width/2, color='IndianRed', label='精确度')
r3 = plt.bar(idx, recalls, width/2, color='Green', label='召回率')
r4 = plt.bar(idx + width/2, f1, width/2, color='Yellow', label='f1')


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2-.1, 1.001*height, '%s' % float(height))

plt.ylabel('%')
plt.xticks(idx, ('贝叶斯', '决策树', 'SVM', '神经网络'))
plt.legend()
plt.title(u"分类器比较")
plt.ylim(0.70,1.0)

autolabel(r1)
autolabel(r2)
autolabel(r3)
autolabel(r4)


plt.show()