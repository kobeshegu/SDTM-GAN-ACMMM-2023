#!/usr/bin/env bash
output_dir="results/flower_lofgan_wavegan_7_shot_baseline"
config="configs/flower_lofgan.yaml"
dataset="flower"

real_dir="${output_dir}/real/img"
fake_dir="${output_dir}/tests/img"


python train.py --conf $config \
--output_dir $output_dir \
--gpu 0 \

echo "evaluating metrics............"
echo $real_dir
echo $fake_dir


python main_metric.py \
    --name $output_dir \
    --dataset $dataset \
    --real_dir $real_dir \
    --fake_dir $fake_dir \
    --gpu 0 \
    --n_sample_test 7 \

real_dir_precision="${output_dir}/real"
fake_dir_precision="${output_dir}/tests"
python Precision_recall.py \
    --name $output_dir \
    --dataset $dataset \
    --real_dir $real_dir_precision \
    --fake_dir $fake_dir_precision \
    --gpu 0 \
    --n_sample_test 7 \


sendemail -f mpyang_ecust@163.com -t mpyang_ecust@163.com -s smtp.163.com -u "dashulan Code Finished!" -o message-content-type=html -o message-charset=utf-8 -xu mpyang_ecust -xp WECCVEFLPVBMVCFK -m "Your code has finished, My Honor to serve you, Sir."


