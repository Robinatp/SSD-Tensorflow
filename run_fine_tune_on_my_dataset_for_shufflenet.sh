DATASET_DIR=`pwd`/tfrecords/
TRAIN_DIR=`pwd`/train_my_dataset_shufflenet_logs/
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=my_dataset \
    --dataset_split_name=train \
    --num_classes=3 \
    --model_name=ssd_300_vgg_my \
    --save_summaries_secs=600 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --log_every_n_steps=1 \
    --max_number_of_steps=100000 \
    --batch_size=2
