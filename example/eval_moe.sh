MODELDIR=${1}
DEVICE=${2}
DATA_FOLDER=${3}

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CUDA_VISIBLE_DEVICES=${DEVICE} fairseq-generate $DATA_FOLDER \
                    --path $MODELDIR \
                    --batch-size 128 --beam 4 --lenpen 0.6 --remove-bpe \
                    --quiet --fp16 --user-dir moe
