from .cic_2019_dataset import CIC2019Dataset


class CIC2019DrDoSMSSQL(CIC2019Dataset):

    def __init__(self, load_preprocessed_data: bool, seed: int) -> None:
        super().__init__(load_preprocessed_data, seed, files=[
            "01-12/DrDoS_MSSQL.csv",
        ])

    def get_dataset_name(self) -> str:
        return "CIC_2019_DrDoS_MSSQL"

    def get_raw_files_directory_name(self) -> str:
        return "CIC_2019"


if __name__ == '__main__':
    dataset = CIC2019DrDoSMSSQL(load_preprocessed_data=False, seed=42)
