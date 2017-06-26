#!/bin/bash

export PYTHONPATH="./src"

# --
# Training w/ dlib alignment (because I don't want to wait for `facenet` alignment)

CASIA_ALIGNED_PATH="/home/bjohnson/data/casia/face_chips-182-40"
RUN=2

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
    --center_loss_alfa 0.9 \
    --gpu-id 1 \
    --gpu_memory_fraction 0.9 | tee log-$RUN

# !! Horizontal flips seem wrong -- faces are asymetrical, that should be exploited
# ~ 14 hours (single IO thread)

# --
# Training w/ MTCNN alignment
# !! This is going to take forever, but should train model to reproduce results

export PYTHONPATH="./src"
for N in {1..4}; do
    python src/align/align_dataset_mtcnn.py \
        /home/bjohnson/data/casia/CASIA-WebFace-clean/ \
        ./data/casia_maxpy_mtcnnpy_182 \
        --image_size 182 --margin 44 --random_order --gpu-id 0 --gpu_memory_fraction 0.2 &
done

export PYTHONPATH="./src"
for N in {1..4}; do
    python src/align/align_dataset_mtcnn.py \
        ./data/lfw/raw \
        ./data/lfw/mtcnnpy_160 \
        --image_size 160 --margin 32 --random_order --gpu-id 1 --gpu_memory_fraction 0.1 &
done


CASIA_ALIGNED_PATH="./data/casia_maxpy_mtcnnpy_182"
RUN=3

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
    --center_loss_alfa 0.9 \
    --gpu-id 0 \
    --gpu_memory_fraction 0.9 | tee log-$RUN
