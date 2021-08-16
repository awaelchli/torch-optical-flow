# RAFT: Recurrent All Pairs Field Transforms for Optical Flow

This folder contains an (inofficial) implementation of the paper

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

The official PyTorch code can be found [here](https://github.com/princeton-vl/RAFT).

## Highlights

- Implemented with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) following best practices
- Fully reprodudible results
- Pretrained checkpoints available
- Configuration files for each training stage
- Optional logging with [Weights and Biases](https://wandb.ai/awaelchli/lightning-raft)


## Requirements

The results were produced with PyTorch 1.9 and PyTorch Lightning 1.4.1.
To install all requirements for training, run:

```bash
pip install -r requirements.txt
```

## Inference

Pretrained models can be downloaded by running
```bash
cd pretrained
./download.sh
```

Generate optical flow from images in a folder:
```bash
python predict.py source destination --checkpoint pretrained/checkpoints/raft-sintel.ckpt
```

Run `python predict.py --help` to display the help page.

## Training

### Download Data
To train or evaluate the RAFT model, download the following datasets:

* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)

By default, our training code expects the datasets to be structured as shown here:

```bash
├── datasets
    ├── MPI-Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
    ├── FlyingChairs
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

The path to the dataset root folderes can be changed in our configuration files under `config/train/default.yaml`.

### Configuration Files

Each training stage has an associated YAML configuration file under `config/train/` containing hyperparameters for model, data and trainer plus settings for logging and checkpointing.
To train using a particular configuration file, run:

```bash
python train.py --config config/train/config-name.yaml
```

### Reproduction of Paper Results

The results in the paper can be reproduced by training the following stages:

```bash
pyton train.py --config config/train/chairs.yaml
pyton train.py --config config/train/things.yaml --restore_weights raft-chairs.ckpt
pyton train.py --config config/train/sintel.yaml --restore_weights raft-things.ckpt
pyton train.py --config config/train/kitti.yaml --restore_weights raft-sintel.ckpt
```

Additional options can be set through the CLI, run `python train.py --help` for more options.

All checkpoints under `pretrained/checkpoints/` were trained using these settings on a single NVIDIA V100 (16 GB) GPU with mixed precision enabled. Each stage takes roughly 15 hours to complete.

The logged results of our runs are available at [Weights and Biases](https://wandb.ai/awaelchli/lightning-raft).


## Evaluation

The trained checkpoints can be evaluated using our configuration files under `config/validate/`, e.g., by running:

```Shell
python validate.py --config config/validate/kitti.yaml
```

## Acknowledgments

The code, including this README file, was adapted from the [official repository](https://github.com/princeton-vl/RAFT).
