from .cic_2018_dataset import CIC2018Dataset


class CIC2018SubSample(CIC2018Dataset):

    def __init__(self, load_preprocessed_data: bool, seed: int) -> None:
        super().__init__(load_preprocessed_data, seed, files=[
            "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
        ])

    def get_dataset_name(self) -> str:
        return "CIC_2018_SubSample"

    def get_raw_files_directory_name(self) -> str:
        return "CIC_2018"


if __name__ == '__main__':
    dataset = CIC2018SubSample(load_preprocessed_data=False, seed=42)
