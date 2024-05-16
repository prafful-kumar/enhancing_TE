# enhancing_TE


## Requirements

- PyTorch version 1.9.0
- torchvision version 0.10.0
- CUDA version 11.1

To install the required packages, run the following command:

```bash
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch
pip install timm==0.4.9
```
![Demo Video](https://github.com/prafful-kumar/enhancing_TE/blob/main/spread_attract_animation.gif)

## Data Preparation

Download the downstream datasets to `../data/*`.

## Model Checkpoints

Download the model checkpoints to `./models/*`.

## Pipeline of Model Selection using Transferability

### Step 1 (optional): Fine-tune pre-trained models with hyper-parameter sweep to obtain ground-truth transferability score

```bash
python finetune_group1.py -m resnet50 -d cifar10
```

### Step 2: Extract features of target data using pre-trained models

```bash
python forward_feature_group1.py -d cifar10
```

### Step 3: Evaluate the transferability of models

```bash
python evaluate_metric_group1_cpu.py -me SA -d cifar10 --type NCTI 
```

### Step 4: Calculate the ranking correlation

```bash
python tw_group1.py --me SA --type NCTI -d sun397
```

### Acknowledgement:

This code repository is developed based on SFDA.
