#!/bin/bash
python train.py \
--name teaser_cat_augment --batch 1 \
--dataroot_sketch /mnt/disk/zw/data/sketch/by_author/cat \
--dataroot_image /mnt/disk/zw/data/image/lmdb/cat --l_image 0.7 \
--g_pretrained /mnt/disk/zw/pretrained/stylegan2-cat/netG.pth \
--d_pretrained /mnt/disk/zw/pretrained/stylegan2-cat/netD.pth \
--max_iter 150000 --disable_eval --diffaug_policy translation \
--resume_iter 128000 \
--checkpoints_dir /mnt/disk/zw/checkpoint/ \
--photosketch_path /mnt/disk/zw/pretrained/photosketch.pth \
2> ./warning/teaser_cat_augment.txt
