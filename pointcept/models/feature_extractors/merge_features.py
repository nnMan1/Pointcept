from .base_feature_extractor import BaseFeatureExtractor
from pointcept.models.builder import build_model, MODELS

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

        x['add_features'] = self.feature_extractor1(x).feat
        outputs = self.feature_extractor2(x)

        return outputs