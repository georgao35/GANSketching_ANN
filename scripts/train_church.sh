#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py \
--name church_jittor --batch 1 \
--dataroot_sketch /mnt/disk/zw/data/sketch/photosketch/gabled_church \
--dataroot_image /mnt/disk/zw/data/image/lmdb/church --l_image 0.7 \
--g_pretrained /mnt/disk/zw/pretrained/stylegan2-church/netG.pth \
--d_pretrained /mnt/disk/zw/pretrained/stylegan2-church/netD.pth \
--resume_iter 12000 \
--max_iter 16001 --disable_eval --diffaug_policy translation \
--photosketch_path /mnt/disk/zw/pretrained/photosketch.pth \
--checkpoints_dir /mnt/disk/zw/checkpoint/ \
--display_freq 1000
