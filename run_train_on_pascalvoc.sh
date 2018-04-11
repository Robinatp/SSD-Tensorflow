DATASET_DIR=`pwd`/tfrecords/
CHECKPOINT_PATH=`pwd`/checkpoints/ssd_300_vgg.ckpt
TRAIN_DIR=`pwd`/train_logs/
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=600 \
    --save_interval_secs=6000 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --log_every_n_steps=1 \
    --batch_size=8
