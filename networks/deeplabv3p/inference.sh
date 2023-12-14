python inference.py \
  --model_dir ./deeplabv3p_mobilenet_tfmodel \
  --data_dir ../../data/data_augmented/images/validation \
  --infer_data_list ../../data/data_augmented/images/validation_images_list.txt \
  --output_dir  ./inference_output \
  --base_architecture mobilenet_v1 
