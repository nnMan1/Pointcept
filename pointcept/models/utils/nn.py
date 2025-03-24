import torch
import torch_scatter
from torch import nn

class SuperpointPooling(nn.Module):

    def __init__(self, pool_function = torch_scatter.scatter_mean):
        super().__init__()

        self.pool_function = pool_function

    def __prepare_seg_indices(self, seg_indices, offset):

        bs = 0
        off = 0

        offs = []

        for be in offset:
            tmp = seg_indices[bs:be]
            _, inverse_indices = torch.unique(tmp, return_inverse=True)
            seg_indices[bs:be] = inverse_indices + off
            off = seg_indices[:be].max() + 1 
            offs.append(off)
            bs = be
    
        return seg_indices, torch.tensor(offs)

    def forward(self, data, keys=['instance', 'segment', 'features']):

         if 'seg_indices' not in data.keys():
               return data

         data['offset_orig'] = data['offset']
         data['seg_indices'], data['offset'] = self.__prepare_seg_indices(data['seg_indices'], data['offset'])

         label_keys = []
         if 'instance' in keys:
            label_keys.append('instance')
            keys.remove('instance')

         if 'segment' in keys:
            label_keys.append('segment')
            keys.remove('segment')

         
         for key in label_keys:
            label = []
            for cls in data['seg_indices'].unique():
               cluster_mask = data['seg_indices'] == cls
          
               unique_labels, counts = torch.unique(data[key][cluster_mask], return_counts=True)
               majority_label = unique_labels[torch.argmax(counts)]
               label.append(majority_label)

            data[key] = torch.stack(label)

         bs = 0
         for be in data['offset']:
            _, new_instance_indices = torch.unique(data['instance'][bs:be], return_inverse=True)
            data['instance'][bs:be] = new_instance_indices
            bs = be

         for key in keys:
            data[key] = self.pool_function(data[key],  data['seg_indices'], dim=0)[0]
        
         return data

class SuperpointUnpooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data, keys=['instance', 'segment']):

        if 'seg_indices' not in data.keys():
            return data

        data['offset'] = data['offset_orig']

        for key in keys:
            data[key] = data[key][data['seg_indices']]
        
        return data
