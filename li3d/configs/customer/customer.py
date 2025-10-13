# 这是一个示例用于列举一些配置中的必要设置
_base_ = ["../_base_/base_config.py"]
# misc custom setting
save_path = "./exp/customer"
batch_size = 2  # bs: total bs in all gpus
num_worker = 16
batch_size_val = 1
empty_cache = False
enable_amp = False
show_memory = True

# model
model = dict(
    type="DefaultRegistration",
    backbone=dict(
        type="MyModel"
    ),
    criteria=[
        dict(type="PoseLoss", w_trans=1.0, w_rot=1.0, reduction="mean", loss_weight=1.0),
        # dict(type="ReconstructionMSE", loss_weight=1.0, reduction='mean'),
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "CustomerDataset"
data_root = r"D:\program\li3D-ML\data\flow_data"
ignore_index = -1
names = [],

data = dict(
    ignore_index=ignore_index,
    names=names,
    # 训练集
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.8,
                 index_valid_keys_key="index_valid_keys"),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5, label_key="transform_matrix"),
            dict(type="RandomScale", scale=[0.9, 1.1], label_key="transform_matrix"),
            dict(type="RandomFlip", p=0.5, label_key="transform_matrix"),
            dict(type="RandomJitter", sigma=0.005, clip=0.02, label_key="transform_matrix"),
            dict(
                type="GridSample",
                grid_size=0.2,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                input_key="coord",
                grid_coord_key="grid_coord",
                index_valid_keys_key="index_valid_keys"
            ),
            dict(
                type="GridSample",
                grid_size=0.2,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                input_key="coord_transformed",
                grid_coord_key="grid_coord_transformed",
                index_valid_keys_key="index_valid_keys_trans"
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                # 下面这些都会保留并转 Tensor（不拼）
                keys=("coord", "grid_coord", "coord_transformed", "grid_coord_transformed", "transform_matrix"),
                offset_keys_dict={"offset": "coord", "offset_trans": "coord_transformed"},
                # 下面两行 = 拼成两个 (N,4) tensor
                orig_keys=("coord", "intensity"),  # → data["orig"] (N,4)
                trans_keys=("coord_transformed", "intensity_transformed"),  # → data["trans"] (N,4)
            )
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    # 验证集
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.2,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                input_key="coord_transformed",
                grid_coord_key="grid_coord_transformed"
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                # 下面这些都会保留并转 Tensor（不拼）
                keys=("coord", "grid_coord", "coord_transformed", "grid_coord_transformed", "transform_matrix"),
                # 下面两行 = 拼成两个 (N,4) tensor
                orig_keys=("coord", "intensity"),  # → data["orig"] (N,4)
                trans_keys=("coord_transformed", "intensity_transformed"),  # → data["trans"] (N,4)
            )
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[]),
        ignore_index=ignore_index, ))

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]
