python train.py \
  --model_dir=./unet_model_0.5 \
  --channel_multiplier=0.5 \
  --train_epochs=10 \
  --epochs_per_eval=10 \
  --batch_size=4 \
  --data_dir=../../data/data_tfrecords/ \
  --max_iter=30000 \
  --initial_learning_rate=1e-3 \
  --end_learning_rate=1e-8 \
