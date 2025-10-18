from .base_feature_extractor import BaseFeatureExtractor
from pointcept.models.builder import build_model, MODELS
import torch

@MODELS.register_module("MergeFeatures")
class MergeFeatures(BaseFeatureExtractor):
    def __init__(self, 
                 model1_config: dict,
                 model2_config: dict,
                 **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor1 = build_model(model1_config)
        self.feature_extractor2 = build_model(model2_config)

    def backbone_modules(self):
        return [self.feature_extractor1, self.feature_extractor2]

    def forward_features(self, x):

        out1 = self.feature_extractor1(x)
        x['add_features'] = out1.feat
        outputs = self.feature_extractor2(x)

        if 'loss' not in outputs:
            outputs['loss'] = torch.tensor(0.0, device=out1.feat.device)

        outputs['loss'] += out1.get('loss', 0.0)

        return outputs