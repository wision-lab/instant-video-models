## Conda Environment

To create the Conda environment, run:
```
conda env create --file environment.yml
```
Then activate the environment with:
```
conda activate stability
```

To update the environment after modifying `environment.yml`, run the following (after activation):
```
conda env update --file environment.yml --prune
```

The file `environment_precise.yml` contains more exact package versions and can be used to reproduce the original development environment. To create an environment based on `environment_precise.yml`, run:
```
conda env create --file environment_precise.yml
```
To generate a new `environment_precise.yml` based on the current environment, run
```
conda env export > environment_precise.yml
```

## Submodules

After cloning this repository, initialize submodules by running:
```
git submodule init
git submodule update
```

## VMAF Setup

After cloning submodules, run the following commands to install VMAF for video quality evaluation. Note - these commands should be run after activating the project Conda environment.
```
cd ./extern/vmaf
make clean
make PYTHON_INTERPRETER=python
pip3 install -r python/requirements.txt
```

Confirm the build was successful by running:
```
./libvmaf/build/tools/vmaf --version
```

Run the script `./scripts/misc/prepare_vmaf_test_data.sh` to download and prepare the data required to run the unit tests in `test_vmaf.py`. 

## Other Setup

Scripts assume that the working directory is on the Python path. To set this up, run the following in Bash (or add it to `.bashrc`):
```
if [[ :$PYTHONPATH: != *:.:* ]]
then
    export PYTHONPATH="$PYTHONPATH:."
fi
```

## Datasets

### VisionSim Depth

Sacha's Blender depth data is located at `pinksquirrel.cs.wisc.edu:/nobackup3/sjungerman/dataset/renders`. To create a symlink to this directory, run the following command:
```
ln -s /nobackup3/sjungerman/dataset/renders data/vision_sim
```

### DAVIS

Run the script `./scripts/datasets/download_davis.sh` to download and unpack the DAVIS dataset.

Run the script `./scripts/datasets/crop_davis.py` to crop the DAVIS dataset to only the annotated sections.

### Need for Speed (NFS)

Run the script `./scripts/datasets/download_nfs.sh` to download and unpack the NFS dataset.

### Local Laplacian NFS

Run the following command (requires a MATLAB installation with the image processing toolbox):
```
matlab -batch 'run("scripts/datasets/generate_nfs_local_laplacian.m")'
```
Comment/uncomment the marked blocks at the top of the file to adjust the strength of the local Laplacian effect.

### SPRING

Download the following files from the [Spring website](https://robust_spring-benchmark.org/):
- `test_frame_left.zip`
- `test_frame_right.zip`
- `fog.zip`
- `frost.zip`
- `rain.zip`
- `snow.zip`
- `spatter.zip`

Place the downloaded files in `data/robust_spring`. Then run `./scripts/datasets/unpack_robust_spring.sh` to unpack the zip files. Note this script deletes the original zip files after unpacking to save space.

### VIPER

Download the following ZIP files from Google Drive:
- Training image sequences (dense frames, compressed):
  - https://drive.google.com/file/d/1-O7vWiMa3mDNFXUoYxE3vkKZQpiDXUCf/view
  - https://drive.google.com/file/d/1alD_fZja9qD7PUnk4AkD6l-jBhlCnzKr/view
  - https://drive.google.com/file/d/19Da-Ac_9KMjexvYGkfjAFowWEGGxMR3I/view
  - https://drive.google.com/file/d/1KZh-z7SeKJDjOG08MWPKX2UBaZ4d-xjF/view
  - https://drive.google.com/file/d/1CeNQ0h1Kr00J45izEYXXhLNghQANRNDG/view
  - https://drive.google.com/file/d/1Vf0MwcgKaz6zgjvqEYlnRbud_lAT4-JK/view
- Training class labels (dense frames):
  - https://drive.google.com/file/d/1lAbmIVuQTLZu4-hNKD20wmGn1SThvFtv/view
  - https://drive.google.com/file/d/1KEDYhQeGQ5qOPY2RoTP1btkWmupdR2Sr/view
  - https://drive.google.com/file/d/1mIdQxrG_UkgHV1HBvMIjL51yaju6v0S3/view
- Validation image sequences (dense frames, compressed):
  - https://drive.google.com/file/d/1951O6Eu-VuMHaL1vJ9V35njcj30GjPiN/view
  - https://drive.google.com/file/d/1OqEjlrx97ThCMlQePEZPSBjqhRqPwOEd/view
  - https://drive.google.com/file/d/1zo5ZKE90N0iE7E_KJU8_FBT0N4ne6knK/view
  - https://drive.google.com/file/d/1QZuPkd_3dRLqZXgwf29gHXtaI8MEbbAS/view
  - https://drive.google.com/file/d/1LgMGWfp_R6hwmMd-zTjrAw5OfflXS01W/view
- Validation class labels (dense frames):
  - https://drive.google.com/file/d/1QN2OSXTDsXPXntNrY-ojDpj-vFWjBZlK/view
  - https://drive.google.com/file/d/1XjXU9qSAvC1JB95ytv15yqyGemJqwK1U/view
  - https://drive.google.com/file/d/1vOgHMuRoPQ0-h-gOKoxXaVIB8ddjEbnp/view

Place the downloaded files in `data/viper`. Then run `./scripts/datasets/unpack_viper.sh` to unpack the zip files. Note this script deletes the original zip files after unpacking to save space.

## Weights

### AdaIn

Download the files `decoder.pth` and `vgg_normalised.pth` from [this GitHub releases page](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0). Place these files in `./weights/adain`. Then run `./scripts/models/prepare_adain_weights.py` to generate a single merged weight file that can be used more easily with our codebase.


### Deeplab

Download the `best_deeplabv3plus_mobilenet_cityscapes_os16.pth` weights using [this link](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) and place them in `weights/deeplab`. Then run `./scripts/models/prepare_deeplab_weights.py` to convert these weights to a format usable with our codebase.

### HDRNet

Download weights using [this link](https://data.csail.mit.edu/graphics/hdrnet/pretrained_models.zip). Extract the contents of the zip file to `weights/hdrnet`. Then run `./scripts/models/prepare_hdrnet_weights.py` to convert these weights to a format usable with our codebase.

### NAFNet

Download the `NAFNet-SIDD-width32.pth` weights using [this Google Drive link](https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view?usp=sharing). Place this file in `./weights/nafnet`. Then run `./scripts/models/prepare_nafnet_weights.py` to convert these weights to a format usable with our codebase.

## Code Style

Format all code using [Black](https://black.readthedocs.io/en/stable/). Use a line limit of 88 characters (the default) for both code and docstrings/comments. To format a file, use the command:
```
black <FILE>
```
Black is installed in the Conda environment.

## Config Naming

Components of a config name should be separated by hyphens, and words within a component should be separated by underscores. For example, `bdd_100k-detr_resnet_50.yml` for a config that uses the `bdd_100k` dataset and the `detr_resnet_50` model.
