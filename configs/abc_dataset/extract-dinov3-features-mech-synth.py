_base_ = ["../_base_/default_runtime.py"]

# Offline DinoV3 patch-feature extraction for MechanicalAssemblySynth
# (train + val splits). Writes fp16 (n_views, 32, 32, 1280) grids per scene
# into HDF5 shards + feature_index.pkl under {save_path}/features.
# Run via scripts/start_feature_extraction.sh (needs /leonardo_scratch bound).

# 4 scenes x 20 views = 80 ViT-H forwards per batch
batch_size = 4
num_worker = 8
mix_prob = 0
empty_cache = False
enable_amp = True  # bf16 backbone forward; engine casts to float before saving fp16
evaluate = False
prefetch_factor = 2
find_unused_parameters = False
weight = None
resume = False
seed = 0

epoch = 1
eval_epoch = 1

# features land on scratch: work quota cannot hold ~110 GB
save_path = "/leonardo_scratch/large/userexternal/vdosljak/dino_features/mech_synth"
shard_size = 50  # scenes per shard (~2.6 GB fp16)

model = dict(
    type="ImageFeatureExtractor",
    model_type="DinoV3",
    model_name="/home/backbones/dinov3-vith16plus-pretrain-lvd1689m",
)

num_classes = 1
classes = {"other": 0, "gear": 0, "nut": 0, "screw": 0, "axe": 0}
class_names = ["other"]

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type="MechanicalAssemblySynth",
        split=["train", "val"],
        data_root="data/segment-assembly-merged-synthetic/data",
        cache=True,
        image_size=(512, 512),
        loop=1,
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("images", "name"),
                offset_keys_dict=dict(image_offset="images"),
            ),
        ],
        test_mode=False,
        classes=classes,
    ),
)

hooks = []

preextractor = dict(type="IMG_Extractor")
