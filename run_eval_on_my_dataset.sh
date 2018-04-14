DATASET_DIR=`pwd`/tfrecords/
TRAIN_DIR=`pwd`/train_my_dataset_logs/
EVAL_DIR=`pwd`/train_my_dataset_logs/eval_logs
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=my_dataset \
    --dataset_split_name=val \
    --num_classes=3 \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=500