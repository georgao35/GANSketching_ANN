python train.py \
--name standing_cat_augment_img --batch 1 \
--dataroot_sketch data/sketch/photosketch/standing_cat \
--dataroot_image data/image/cat --l_image 0.7 \
--g_pretrained pretrained/stylegan2-cat/netG.pth \
--d_pretrained pretrained/stylegan2-cat/netD.pth \
--max_iter 200000 --disable_eval --diffaug_policy translation \
--checkpoints_dir checkpoint/ \
--photosketch_path pretrained/photosketch.pth \
--display_freq 2000 \
