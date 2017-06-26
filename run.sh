export PYTHONPATH="./src:$PYTHONPATH"

# --
# Align images

for N in {1..4}; do 
    python src/align/align_dataset_mtcnn.py ./data/lfw/raw ./data/lfw/mtcnnpy_160 \
        --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.20 & 
done

# --
# Embeddings

# native facenet aligned
python src/validate_on_lfw.py ./data/lfw/mtcnnpy_160/ ./models/20170512-110547
# Accuracy: 0.992+-0.003
# Validation rate: 0.97467+-0.01477 @ FAR=0.00133
# Area Under Curve (AUC): 1.000
# Equal Error Rate (EER): 0.007

# dlib aligned
python src/validate_on_lfw.py /home/bjohnson/data/lfw/train/lfw-aligned-160/ ./models/20170512-110547 --lfw_file_ext jpg

# --
# Simple embeddings

find ./data/lfw/mtcnnpy_160/ -type f | fgrep png | ./src/simple-run.py --model ./models/20170512-110547 > feats


# --
# Instagram face embeddings

function get_instagram_faces {
    find /home/bjohnson/data/instagram/trickle/images/face_chips/ -type f | fgrep jpg
}

OUTPATH="/home/bjohnson/data/instagram/trickle/images/feats.facenet"
get_instagram_faces | ./src/simple-run.py --model ./models/20170512-110547 > $OUTPATH

SMALL_OUTPATH="/home/bjohnson/data/instagram/trickle/images/feats-60.facenet"
get_instagram_faces |\
    ./src/simple-run.py --model ./models/20170512-110547 --bottleneck-dim 60 --gpu 1 > $SMALL_OUTPATH