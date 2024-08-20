import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import json
import numpy as np
from pointcept.utils.visualization import colors

logs = ['/home/exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_grid_size_0_3/result0.3/test0.3.log',
        '/home/exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_normals_old/result_0_3/test.log',
        '/home/exp/fuselage/semseg-spunet-v1m1-0-base_lr_split_grouping3/result/test.log']

model_names = ['no_normals', 'normas embeded as feature', 'direct normal embedding']

title = ''

metrics =  {'mIoU': [], 'mAcc': [], 'allAcc': []}

for log in logs:
    v = open(log).readlines()[-6].split(': ')[-1]
    m, v = v.split(' ')
    m = m.split('/')
    v = (float(val) for val in v.split('/'))
    
    for metr, val in zip(m, v):
        if metr not in metrics.keys():
            metrics[metr] = []

        metrics[metr].append(val)


ax = plt.subplot(111)
w = 1 / (len(model_names) + 2)

for i, name in enumerate(model_names):
    coords =  np.arange(len(metrics)) - ((len(model_names) / 2) - i - 0.5) * w
    vals = [m[i] for m in metrics.values()]
    ax.bar(coords, vals, width=w, color=colors[i], align='center', label=name)    

ax.autoscale(tight=False)
ax.set_xticks(ticks=np.arange(len(metrics)), labels=metrics.keys())
ax.legend(loc="lower right")
ax.set_title(title)
# plt.show()
plt.savefig('metrics.png')
