python train.py \
  --model_dir ./deeplabv3p_mobilenet_tfmodel \
  --epochs_per_eval 1 \
  --batch_size 4 \
  --data_dir ../../data/data_tfrecords/ \
  --base_architecture  mobilenet_v1 \
  --pre_trained_model ../../pretrained_models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt
