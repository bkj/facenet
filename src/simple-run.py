#!/usr/bin/env python

"""
    simple-run.py
"""

from __future__ import division

import re
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image

from keras import backend as K
def limit_mem():
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    cfg.gpu_options.visible_device_list="0"
    K.set_session(K.tf.Session(config=cfg))

limit_mem()

# --
# Helpers

def load_model(model_dir):
    model_dir = os.path.expanduser(model_dir)
    
    files = os.listdir(model_dir)
    
    meta_file = [s for s in files if s.endswith('.meta')][0]
    
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    
    saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir, ckpt_file))
    
    return (
        tf.get_default_graph().get_tensor_by_name("input:0"),
        tf.get_default_graph().get_tensor_by_name("phase_train:0"),
        tf.get_default_graph().get_tensor_by_name("embeddings:0"),
    )


def load_img(path, bottleneck_dim=None, target_dim=160):
    # Load
    img = Image.open(path).convert('RGB')
    
    # Resize
    if bottleneck_dim > 0:
        img = img.resize((bottleneck_dim, bottleneck_dim))
    
    if (img.width != target_dim) or (img.height != target_dim):
        img = img.resize((target_dim, target_dim), Image.BILINEAR)
        
    img = np.array(img)
    
    # Whiten
    img = img - img.mean()
    img = img / np.maximum(np.std(img), 1.0 / np.sqrt(img.size))
    
    return img

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./models/20170512-110547')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--bottleneck-dim', type=int, default=-1)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    paths = [line.strip() for line in sys.stdin]
    
    n_batches = int(np.ceil(len(paths) / args.batch_size))
    print >> sys.stderr, "n_batches = %d" % n_batches
    
    with tf.Graph().as_default():
        
        # Set TF config
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.visible_device_list=str(args.gpu)
        
        with tf.Session(config=cfg) as sess:
            img_, phase_, emb_ = load_model(args.model)
            all_embs = []
            for i,paths_batch in enumerate(np.array_split(paths, n_batches)):
                print >> sys.stderr, "%d images processed" % (i * args.batch_size)
                embs = sess.run(emb_, feed_dict={
                    img_ : np.array(map(lambda x: load_img(x, bottleneck_dim=args.bottleneck_dim), paths_batch)), 
                    phase_ : False
                })
                
                for p,e in zip(paths_batch, embs):
                    print '\t'.join((
                        p,
                        '\t'.join(map(str, e))
                    ))

