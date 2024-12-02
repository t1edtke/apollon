from .cic_2017_dataset import CIC2017Dataset


class CIC2017SubSample(CIC2017Dataset):

    def __init__(self, load_preprocessed_data: bool, seed: int) -> None:
        super().__init__(load_preprocessed_data, seed, files=[
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
        ])

    def get_dataset_name(self) -> str:
        return "CIC_2017_SubSample"

    def get_raw_files_directory_name(self) -> str:
        return "CIC_2017"


if __name__ == '__main__':
    dataset = CIC2017SubSample(load_preprocessed_data=False, seed=42)
