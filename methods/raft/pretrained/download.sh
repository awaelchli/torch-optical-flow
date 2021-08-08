#!/bin/bash
# our reproduced checkpoints
wget https://torch-optical-flow.s3.eu-central-1.amazonaws.com/checkpoints/raft/chairs-epoch%3D43-step%3D119999.ckpt -O checkpoints/raft-chairs.ckpt
wget https://torch-optical-flow.s3.eu-central-1.amazonaws.com/checkpoints/raft/things-epoch%3D7-step%3D119999.ckpt -O checkpoints/raft-things.ckpt
