## Neural Machine Translation 

Please note that, this working example is build upon the the [THOR](https://github.com/microsoft/Stochastic-Mixture-of-Experts/tree/main) repo. 

### Pre-processing

Please refer to the [Transformer-Clinic](https://github.com/LiyuanLucasLiu/Transformer-Clinic/blob/master/pre-process/wmt14en-de.sh) repo for data preparation. 


### Environment

We recommend to use the docker image `nvcr.io/nvidia/pytorch:22.02-py3` for this example. 

### Training and EVALUATION

```
# for model training, the resulting model will be saved to `output-${NUM_OF_EXPERTS}`.
bash run_moe.sh ${NUM_OF_EXPERTS} ${ROUTER} ${PATH_TO_DATA}

# for model inference, the script will load model weights from `output-${NUM_OF_EXPERTS}/checkpoint_best.pt`. 
bash eval_moe.sh output-${NUM_OF_EXPERTS}/checkpoint_best.pt ${GPU_ID} ${PATH_TO_DATA}
```
