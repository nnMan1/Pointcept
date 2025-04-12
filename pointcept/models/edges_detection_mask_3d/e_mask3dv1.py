from torch import nn
from ..losses import build_criteria
from ..builder import MODELS
from ..instance_segmentation_transformer_base.nn import SuperpointPooling, SuperpointUnpooling

@MODELS.register_module()
class EMask3DEncoder(InstSegTransformerEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_features_head = nn.Linear(
            in_features=kwargs['backbone_out_channels'],
            out_features=self.out_channels,
            bias=True
        )
    
    def forward(self, data):
        tmp = super().forward(data)

        features = self.mask_features_head(tmp['features'][0])

        return {
            'features': features, 
            'aux': tmp['features'][1],
        }

@MODELS.register_module('EMask3D')
class EMask3D(InstanceSegmentationTransformerBase):
    def __init__(self, **kwargs):

        hidden_dim = kwargs.pop('hidden_dim', 128)
        self.num_query = kwargs.pop('num_query', 100)

        self.edges_criteria = build_criteria(kwargs.pop('edges_criteria'))
        super().__init__(**kwargs)

        self.edges_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
)

    def loss(self, input_dict):
        pass

    def forward(self, data):
        data.update(self.encoder(data))

        # print(x.shape)
        # print(len(aux))

        print(data['features'].shape)
        exit(0)

        pass

