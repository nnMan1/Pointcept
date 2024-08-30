from pointcept.models.sparse_unet.mink_unet_v2 import Res16UNet34C

model = Res16UNet34C(6, 20, {
    'dialations': [ 1, 1, 1, 1 ],
    'conv1_kernel_size': 5,
    'bn_momentum': 0.02,
}, True)

print(model)