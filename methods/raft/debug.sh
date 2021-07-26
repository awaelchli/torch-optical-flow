#!/bin/bash

# python train.py --config config/chairs.yaml --trainer.gpus 1 --data.root_chairs ~/optical-flow-datasets/FlyingChairs/data
python train.py --config config/things.yaml --trainer.gpus 1 --data.root_things ~/optical-flow-datasets/FlyingThings3D --data.root_sintel ~/optical-flow-datasets/MPI-Sintel  --restore_ckpt pretrained/models/raft-chairs.pth
