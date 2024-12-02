# apollon

---

The Jupyter Notebook located at `main/evaluate.ipynb` contains the code used to generate the evaluations presented in
the associated research work. Within the notebook, you can customize the dataset and the adversarial attack used for analysis.

## Datasets
Place dataset files in the following directories:
* `live/data_raw/CIC-2017/`
* `live/data_raw/CIC-2018/`
* `live/data_raw/CIC-2019/`

Datasets available at:
* [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)
* [CIC-IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html)
* [CIC-DDoS-2019](https://www.unb.ca/cic/datasets/ddos-2019.html)

## Environment
Create conda environment from environment.yml file:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate apollon
```

Install Deep-Forest from GitHub via setup.py:
```bash
git clone https://github.com/t1edtke/Deep-Forest.git
cd Deep-Forest
python setup.py install
```
