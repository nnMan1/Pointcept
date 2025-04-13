import torch_scatter
from torch import nn
from .utils import *

class SuperpointPooling(nn.Module):

    def __init__(self, pool_function = torch_scatter.scatter_mean, instance_ignor_index=-1):
        super().__init__()

        self.pool_function = pool_function
        self.instance_ignor_index = instance_ignor_index

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

         data['offset_orig'] = data['offset']

         if 'seg_indices' not in data.keys():
            data['seg_indices'] = torch.arange(data['coord'].shape[0], device=data['coord'].device)
            return data
         else:
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
            instances = data['instance'][bs:be]
            non_ignore_mask = instances != self.instance_ignor_index
            if non_ignore_mask.sum() == 0:
                continue

            _, new_instance_indices = torch.unique(instances[non_ignore_mask], return_inverse=True)
            data['instance'][bs:be][non_ignore_mask] = new_instance_indices
            bs = be

         for key in keys:
            data[key] = torch_scatter.scatter_mean(data[key],  data['seg_indices'], dim=0)
        
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

class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)

        if output_use_norm:
            layers.append(norm(output_dim))

        if output_use_activation:
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for (_, param) in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        output = self.layers(x)
        return output
   
class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
    
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )

class CrossAttentionLayer(nn.Module):
    
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        query,
        key,
        value,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        query_ = self.multihead_attn(
            query=self.with_pos_embed(query, query_pos),
            key=self.with_pos_embed(key, pos),
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        query_ = query + self.dropout(query_)
        query_ = self.norm(query_)
        return query_

    def forward_pre(
        self,
        query,
        key,
        value,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        query = self.norm(query)

        query_ = self.multihead_attn(
            query=self.with_pos_embed(query, query_pos),
            key=self.with_pos_embed(key, pos),
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        query = query + self.dropout(query_)

        return query

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                query,
                key,
                value,
                attn_mask=attn_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
        
        return self.forward_post(
                query,
                key,
                value,
                attn_mask=attn_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )

class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

