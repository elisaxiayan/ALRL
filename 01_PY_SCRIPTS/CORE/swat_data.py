import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# SWaT 六个物理 Stage
# ============================================================

STAGE_DEVICES = {

    1: [
        "FIT101", "LIT101",
        "MV101", "P101", "P102"
    ],

    2: [
        "AIT201", "AIT202", "AIT203", "FIT201",
        "MV201",
        "P201", "P202", "P203",
        "P204", "P205", "P206"
    ],

    3: [
        "DPIT301", "FIT301", "LIT301",
        "MV301", "MV302", "MV303", "MV304",
        "P301", "P302"
    ],

    4: [
        "AIT401", "AIT402", "FIT401", "LIT401",
        "P401", "P402", "P403", "P404",
        "UV401"
    ],

    5: [
        "AIT501", "AIT502", "AIT503", "AIT504",
        "FIT501", "FIT502", "FIT503", "FIT504",
        "PIT501", "PIT502", "PIT503",
        "P501", "P502"
    ],

    6: [
        "FIT601",
        "P601", "P602", "P603"
    ]
}


# ============================================================
# 根据column名字自动把binary features分到6个Stage
# ============================================================

def get_stage_feature_columns(df):

    metadata = {
        "Timestamp",
        "Attack_Type",
        "Fold"
    }

    binary_columns = [
        col
        for col in df.columns
        if col not in metadata
    ]

    stage_columns = {}

    for stage, devices in STAGE_DEVICES.items():

        cols = []

        for col in binary_columns:

            for device in devices:

                if col.startswith(device + "_"):

                    cols.append(col)
                    break

        stage_columns[stage] = cols


    # 安全检查
    assigned = []

    for cols in stage_columns.values():
        assigned.extend(cols)

    missing = (
        set(binary_columns)
        -
        set(assigned)
    )

    if len(missing) > 0:

        raise ValueError(
            f"有binary features没有分Stage: {missing}"
        )

    if len(assigned) != len(set(assigned)):

        raise ValueError(
            "存在重复分配的binary feature"
        )

    return stage_columns


# ============================================================
# Dataset
#
# 每一条样本返回：
#
# [Stage1 tensor,
#  Stage2 tensor,
#  ...
#  Stage6 tensor],
# label
# ============================================================

class SWaTBinaryDataset(Dataset):

    def __init__(
        self,
        df,
        stage_columns,
        class_to_id
    ):

        self.stage_data = []

        for stage in range(1, 7):

            cols = stage_columns[stage]

            tensor = torch.tensor(
                df[cols].values,
                dtype=torch.float32
            )

            self.stage_data.append(
                tensor
            )


        labels = (
            df["Attack_Type"]
            .map(class_to_id)
            .values
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, index):

        stage_inputs = [
            stage[index]
            for stage in self.stage_data
        ]

        return (
            stage_inputs,
            self.labels[index]
        )