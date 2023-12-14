python inference.py \
  --data_dir=../../data/data_augmented/images/validation/ \
  --infer_data_list=../../data/data_augmented/images/validation_images_list.txt \
  --model_dir=./unet_model_0.5 \
  --channel_multiplier=0.5 \
  --output_dir=./data/inference_output_0.5
