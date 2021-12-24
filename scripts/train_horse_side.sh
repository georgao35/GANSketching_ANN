#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py \
--name horse_side_augment --batch 1 \
--dataroot_sketch /mnt/disk/zw/data/sketch/photosketch/horse_on_a_side \
--dataroot_image /mnt/disk/zw/data/image/lmdb/horse --l_image 0.7 \
--g_pretrained /mnt/disk/zw/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /mnt/disk/zw/pretrained/stylegan2-horse/netD.pth \
--max_iter 76001 --diffaug_policy translation --disable_eval \
--photosketch_path /mnt/disk/zw/pretrained/photosketch.pth \
--checkpoints_dir /mnt/disk/zw/checkpoint \
--display_freq 2000 \
--resume_iter 50000 \
2> ./warning/horse_side_augment.txt

CUDA_VISIBLE_DEVICES=3 python train.py \
--name horse_side_no_augment --batch 1 \
--dataroot_sketch /mnt/disk/zw/data/sketch/photosketch/horse_on_a_side \
--dataroot_image /mnt/disk/zw/data/image/lmdb/horse --l_image 0.7 \
--g_pretrained /mnt/disk/zw/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /mnt/disk/zw/pretrained/stylegan2-horse/netD.pth \
--max_iter 76001 --disable_eval \
--photosketch_path /mnt/disk/zw/pretrained/photosketch.pth \
--checkpoints_dir /mnt/disk/zw/checkpoint \
--display_freq 2000 \
--resume_iter 50000 \
2> ./warning/horse_side_no_augment.txt
