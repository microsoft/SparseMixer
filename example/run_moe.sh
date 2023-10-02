EXPERTS=${1:-"4"}
ROUTER=${2:-"SparseMixer"}
DATA_FOLDER=${3:-"/mnt/azstorage/wmt14_en_de_joined_dict/"}

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

pip install --upgrade numpy scipy
pip install -r requirements.txt
pip install --no-deps fairseq==0.12.1

ARGS="--num-experts ${EXPERTS} --jitter-eps 0.1 --load-balancing-alpha 0.01 --router ${ROUTER}"
echo $ARGS 

OUTPUT_FOLDER=./output-${EXPERTS}

mkdir -p $OUTPUT_FOLDER

touch $OUTPUT_FOLDER/train.log

cp $OUTPUT_FOLDER/train.log ./train.log

GPUCT=$(nvidia-smi --list-gpus | wc -l)
UPDATE_FREQ=$((16/${GPUCT}))
echo $UPDATE_FREQ

unset RANK
fairseq-train ${DATA_FOLDER} --num-workers 8 --ddp-backend=no_c10d \
      --user-dir moe --task translation_switch --load-balancing \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
      --arch switch_transformer --share-all-embeddings \
      --source-lang en --target-lang de \
      --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
      --warmup-init-lr 1e-07 --warmup-updates 8000 --max-update 400000 \
      --lr 7e-4 --max-tokens 8192 --update-freq ${UPDATE_FREQ} \
      --weight-decay 0 --dropout 0.1 --activation-dropout 0.1 --attention-dropout 0.1 \
      --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
      --log-format simple --log-interval 100 \
      --skip-invalid-size-inputs-valid-test \
      --best-checkpoint-metric loss --save-interval 10 \
      --encoder-layers 6 --decoder-layers 6 \
      --save-dir $OUTPUT_FOLDER $ARGS 2>&1 | tee -a ./train.log

cp ./train.log $OUTPUT_FOLDER/train.log 

sleep 15m
