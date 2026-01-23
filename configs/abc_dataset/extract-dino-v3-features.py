_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 2 # bs: total bs in all gpus
num_worker = 2
mix_prob = 0
empty_cache = True
enable_amp = False
evaluate = True

num_classes = 1
fts_sizes = 128
dim_feedforward=1024
segment_ignore_index = (-1, )

# model settings
model = dict(
    type="ImageFeatureExtractor",
    model_type="DinoV2",
    model_name="facebook/dinov2-small",
)

epoch = 1
eval_epoch = 1

classes=dict({
            'other': 0,
            'nut': 0,
            'screw': 0
        })

class_names = ["other"]

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=['class_names'],
    train=dict(
        type='HDF5_Dataset',
        split='val',
        data_root='data/data/processed/abc_dataset',
        load_images=True,
        transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "images",
                        "name"
                    ),
                    feat_keys=("images"),
                    offset_keys_dict=dict(
                        image_offset="images",
                    ),
                ),
            ],
            test_mode=False,
            classes=classes,
    )
)

hooks = [
    dict(type="CheckpointLoader", keywords=["module.", "module.encoder.backbone.input_conv.0"], replacement=["module.encoder.backbone.", "dummy"]),
]

preextractor = dict(
    type="IMG_Extractor"
)

shard_size = 1000  