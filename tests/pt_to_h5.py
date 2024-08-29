import os
import os.path as osp
import torch
import h5py

files_in = '/home/data/ABCDataset/scans'
files_out = '/home/data/ABCDataset/scanns'

files_list = os.listdir(files_in)

for file in files_list:
    try:
        data = torch.load(osp.join(files_in, file))
        data = {
            'coord': data.pos.numpy(),
            'instance': data.y.numpy(),
            'id': data.assembly.split('/')[-1],
            't': data.t.numpy(),
            'r': data.r.numpy(),
            'hlw': data.lenghts.numpy()
        }

        f = h5py.File(osp.join(files_out, f'{file.split(".")[0]}.h5'), 'w')

        for key, value in data.items():
            dset = f.create_dataset(key, data=value)
    except:
        pass
