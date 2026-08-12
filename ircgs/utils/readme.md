undistort_with_3dgs_convert.py去畸变：python ircgs/utils/undistort_with_3dgs_convert.py \
  --dataset-path data/cl-splats/WAT/breville \
  --convert-script /你的3DGS官方仓库/convert.py \
  --overwrite
undistort_with_3dgs_convert.py去畸变：python ircgs/utils/undistort_with_3dgs_convert.py \
  --dataset-path data/cl-splats/WAT/car_resized \
  --convert-script gaussian-splatting/convert.py \
  --overwrite
  create_trian_test:划分训练集和测试集:python -m ircgs.utils.create_train_test_split \
  --dataset-path /root/autodl-tmp/cl-splats-reproduction-main/data/cl-splats/WAT/breville \
  --test-ratio 0.125 \
  --split-seed 42 \
  --overwrite
一共有street/spa/ninja/mac/living_room/kitchen/grill_resized/dyson/community/car_resized/breville这几个场景


sam+视频切分：python ircgs/utils/prepare_clsplats_object_masks.py \
  --dataset_root /root/autodl-tmp/cl-splats-reproduction-main/data/cl-splats/WAT/spa \
  --deva_root /root/autodl-tmp/cl-splats-reproduction-main/gaussian-grouping-main/Tracking-Anything-with-DEVA \
  --offload_image_features_to_cpu \
  --sam_variant original


单个图片切分：
python ircgs/utils/run_sam_on_folder.py \
  --input_path /root/autodl-tmp/cl-splats-reproduction-main/tmp-sam-test \
  --deva_root /root/autodl-tmp/cl-splats-reproduction-main/gaussian-grouping-main/Tracking-Anything-with-DEVA

你的图片目录/
├── 原图...
└── sam_outputs/
    ├── object_mask/
    └── object_mask_color/
