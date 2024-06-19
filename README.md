# INR-PCC

Official Implementation of "Point Cloud Compression with Implicit Neural Representations: A Unified Framework"

## Requirements

- pytorch 1.9.0
- open3d 0.18.0
- deepCABAC 0.1.0: https://github.com/fraunhoferhhi/DeepCABAC
- tmc3 v23.0-rc2: https://github.com/MPEGGroup/mpeg-pcc-tmc13
- pc_error: https://github.com/minhkstn/mpeg-pcc-dmetric
- 8iVFB dataset: https://plenodb.jpeg.org/pc/8ilabs

The two softwares tmc3 and pc_error are already contained in this repo.

## Usage

```
sudo chmod 777 tmc3 pc_error
```

### Geometry compression

```
python train.py --cloud_path=cloud/loot_vox10_1200.ply --exp_dir=exp/geometry
python test.py --cloud_path=cloud/loot_vox10_1200.ply --exp_dir=exp/geometry
```

### Attribute compression

```
python train.py --cloud_path=cloud/loot_vox10_1200.ply --geometry_path=exp/geometry/cloud.ply --exp_dir=exp/attribute
python test.py --cloud_path=cloud/loot_vox10_1200.ply --geometry_path=exp/geometry/cloud.ply --exp_dir=exp/attribute
```

### Baseline (G-PCC)

```
python baseline.py --cloud_path=cloud/loot_vox10_1200.ply --result_dir=gpcc --pqs=0.75
python baseline.py --cloud_path=cloud/loot_vox10_1200.ply --result_dir=gpcc --encode_colors --pqs=0.75 --qp=34
```