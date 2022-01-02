#!/bin/bash
python train.py \
--name church_jt_aug --batch 1 \
--dataroot_sketch data/sketch/photosketch/gabled_church \
--dataroot_image data/image/lmdb/church --l_image 0.7 \
--g_pretrained pretrained/stylegan2-church/netG.pth \
--d_pretrained pretrained/stylegan2-church/netD.pth \
--resume_iter 12000 \
--max_iter 16001 --disable_eval --diffaug_policy translation \
--photosketch_path pretrained/photosketch.pth \
--checkpoints_dir checkpoint/ \
--display_freq 1000
