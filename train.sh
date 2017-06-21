#!/bin/bash

export PYTHONPATH="./src"

CASIA_PATH="/home/bjohnson/data/casia/CASIA-WebFace-clean"
CASIA_ALIGNED_PATH="./data/CASIA-WebFace-clean-aligned"

# Align
python src/align/align_dataset_mtcnn.py $CASIA_PATH $CASIA_ALIGNED_PATH \
    --image_size 182 --margin 44 --gpu-id 0 --gpu_memory_fraction 0.8

# --
# Training w/ dlib alignment (because I don't want to wait for `facenet` alignment)

CASIA_ALIGNED_PATH="/home/bjohnson/data/casia/face_chips-160"

python src/train_softmax.py \
    --logs_base_dir ./logs \
    --models_base_dir ./models \
    --data_dir $CASIA_ALIGNED_PATH \
    --image_size 160 \
    --model_def models.inception_resnet_v1 \
    --optimizer RMSPROP \
    --learning_rate -1 \
    --max_nrof_epochs 80 \
    --keep_probability 0.8 \
    --random_crop \
    --random_flip \
    --learning_rate_schedule_file ./data/learning_rate_schedule_classifier_casia.txt \
    --weight_decay 5e-5 \
    --center_loss_factor 1e-2 \
    --center_loss_alfa 0.9

# !! Horizontal flips seem wrong -- faces are asymetrical, that should be exploited