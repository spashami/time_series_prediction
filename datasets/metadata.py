from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    dir_path: str
    train_name: str
    validation_name: str
    test_name: str
    columns_name: str


dataset_metadata = {
    "sphere_decay_CFD": DatasetMetadata(
        dir_path="datasets/SphereDecay/",
        train_name="05SphereDecayCFD_V3",
        validation_name="03SphereDecayCFD_V3",
        test_name="01SphereDecayCFD_V3",
        columns_name="columns_name.txt",
    )
}
