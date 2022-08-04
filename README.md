# Keypoint Message Passing

This repository hosts the code for our paper [Keypoint Message Passing for Video-based Person Re-Identification](https://arxiv.org/abs/2111.08279).

`Working in progress...`



## Build Environment

1. Make sure [conda](https://www.anaconda.com/products/individual) is installed.

2. Create environment from file:

```bash
conda env create -f environment.yml
```

3. Install [torchreid](https://github.com/KaiyangZhou/deep-person-reid)

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid/
pip install -r requirements.txt
python setup.py develop
```



## Prepare Dataset

1. Download [MARS dataset](http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html) and [keypoints](https://drive.google.com/file/d/16M0Y8yCgMgqkSeJtlh6gBoVfbZKvuEWE/view?usp=sharing).

2. Organize the file tree as below:
```
PoseTrackReID
└── data
    └── mars
        ├── info/
        ├── bbox_train/
        ├── bbox_test/
        ├── bbox_train_keypoints/
        ├── bbox_test_keypoints/
```



## Run

```bash
# training
CUDA_VISIBLE_DEVICES=0 python scripts/trainval_vginteract.py --cfg_file <prefix>/cfg.yaml --data.save_dir logs/<version_number>/ --data.sources ['marspose'] --data.targets ['marspose'] --train.max_epoch <epoch_number>

# testing
CUDA_VISIBLE_DEVICES=0 python scripts/trainval_vginteract.py --cfg_file logs/<version_number>/<time_stamp_and_machine_name>/cfg.yaml --model.resume logs/<version_number>/<time_stamp_and_machine_name>/model/model.pth.tar-<epoch_number> --test.evaluate
```
Note: `cfg.yaml` contains the default hyper-parameters. The following flags override the defaut hyper-params.



## Citation

```bibtex
@inproceedings{chen2021keypoint,
  title={Keypoint Message Passing for Video-based Person Re-Identification},
  author={Chen, Di and Doering, Andreas and Zhang, Shanshan and Yang, Jian and Gall, Juergen and Schiele, Bernt},
  booktitle={AAAI},
  year={2022}
}
```

