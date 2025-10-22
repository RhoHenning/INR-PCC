# INR-PCC

Official implementation of paper [Point Cloud Compression with Implicit Neural Representations: A Unified Framework](https://ieeexplore.ieee.org/abstract/document/10681880), IEEE/CIC ICCC 2024.

**This work has been extended to [NeRC<sup>3</sup>](https://github.com/RhoHenning/NeRC3), which further investigates the methodological details and explores multiple solutions to support dynamic point cloud compression.**

## Requirements

- pytorch 1.9.0
- open3d 0.18.0
- deepCABAC 0.1.0: https://github.com/fraunhoferhhi/DeepCABAC
- tmc3 v23.0-rc2: https://github.com/MPEGGroup/mpeg-pcc-tmc13
- pc_error: https://github.com/minhkstn/mpeg-pcc-dmetric
- 8iVFB dataset: https://plenodb.jpeg.org/pc/8ilabs

## Usage

The two softwares tmc3 and pc_error are already contained in this repo. Use the following command to change file permissions.

```
sudo chmod 777 tmc3 pc_error
```

### Geometry compression

Make sure that the configuration file `config.yaml` is available in `exp/geometry`. Then use the following commands for geometry compression.

```
python train.py --cloud_path=loot_vox10_1200.ply --exp_dir=exp/geometry
python test.py --cloud_path=loot_vox10_1200.ply --exp_dir=exp/geometry
```

The results can be found in `exp/geometry/result.csv`.

### Attribute compression

Make sure that the configuration file `config.yaml` is available in `exp/attribute`. Then use the following commands for attribute compression.

```
python train.py --cloud_path=loot_vox10_1200.ply --geometry_path=exp/geometry/cloud.ply --exp_dir=exp/attribute
python test.py --cloud_path=loot_vox10_1200.ply --geometry_path=exp/geometry/cloud.ply --exp_dir=exp/attribute
```

The results can be found in `exp/attribute/result.csv`.

### Baseline (G-PCC)

This repo supports compression using G-PCC, by directly calling the tmc3 software with predefined configurations.

Use the following command for geometry compression by G-PCC.

```
python baseline.py --cloud_path=loot_vox10_1200.ply --result_dir=gpcc --pqs=0.75
```

Use the following command for joint geometry and attribute compression by G-PCC.

```
python baseline.py --cloud_path=loot_vox10_1200.ply --result_dir=gpcc --encode_colors --pqs=0.75 --qp=34
```

The results can be found in `gpcc/result.csv`.

## Citation

If you find this work helpful in your research, please cite as follows:

```
@inproceedings{ruan2024inrpcc,
  title={Point Cloud Compression with Implicit Neural Representations: A Unified Framework},
  author={Ruan, Hongning and Shao, Yulin and Yang, Qianqian and Zhao, Liang and Niyato, Dusit},
  booktitle={2024 IEEE/CIC International Conference on Communications in China (ICCC)},
  pages={1709--1714},
  year={2024}
}
```