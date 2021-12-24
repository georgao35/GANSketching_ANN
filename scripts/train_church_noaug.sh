#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
--name church_jt_noaug_grad --batch 1 \
--dataroot_sketch /mnt/disk/zw/data/sketch/photosketch/gabled_church \
--dataroot_image /mnt/disk/zw/data/image/lmdb/church --l_image 0.7 \
--g_pretrained /mnt/disk/zw/pretrained/stylegan2-church/netG.pth \
--d_pretrained /mnt/disk/zw/pretrained/stylegan2-church/netD.pth \
--max_iter 14001 --disable_eval \
--resume_iter 6000 \
--photosketch_path /mnt/disk/zw/pretrained/photosketch.pth \
--checkpoints_dir /mnt/disk/zw/checkpoint/ \
--display_freq 2000 \
