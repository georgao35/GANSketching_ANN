CUDA_VISIBLE_DEVICES=0 python train.py \
--name standing_cat_no_aug --batch 1 \
--dataroot_sketch /mnt/disk/zw/data/sketch/photosketch/standing_cat \
--dataroot_image /mnt/disk/zw/data/image/lmdb/cat --l_image 0.7 \
--g_pretrained /mnt/disk/zw/pretrained/stylegan2-cat/netG.pth \
--d_pretrained /mnt/disk/zw/pretrained/stylegan2-cat/netD.pth \
--max_iter 300000 --disable_eval \
--resume_iter 198000 \
--checkpoints_dir /mnt/disk/zw/checkpoint/ \
--photosketch_path /mnt/disk/zw/pretrained/photosketch.pth \
--display_freq 2000 \
2> ./warning/standing_cat_no_aug.txt
