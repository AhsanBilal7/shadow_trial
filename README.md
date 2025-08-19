## Requirement
* Python 3.7
* Pytorch 1.7
* CUDA 11.1
```bash
pip install -r requirements.txt
```

## Datasets
* ISTD [[link]](https://github.com/DeepInsight-PCALab/ST-CGAN)  
* ISTD+ [[link]](https://github.com/cvlab-stonybrook/SID)

## Test
You can directly test the performance of the pre-trained model as follows
1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the `test.py` 
```python
input_dir # shadow image input path -- Line 27
weights # pretrained model path -- Line 31
```
2. Test the model
```python
python test.py --save_images
```
You can check the output in `./results`.

## Train
1. Download datasets and set the following structure
```
|-- ISTD_Dataset
    |-- train
        |-- train_A # shadow image
        |-- train_B # shadow mask
        |-- train_C # shadow-free GT
    |-- test
        |-- test_A # shadow image
        |-- test_B # shadow mask
        |-- test_C # shadow-free GT
```
2. You need to modify the following terms in `option.py`
```python
train_dir  # training set path
val_dir   # testing set path
gpu: 0 # Our model can be trained using a single RTX A5000 GPU. You can also train the model using multiple GPUs by adding more GPU ids in it.
```
3. Train the network
If you want to train the network on 256X256 images:
```python
python train.py --warmup --win_size 8 --train_ps 256
```
or you want to train on original resolution, e.g., 480X640 for ISTD:
```python
python train.py --warmup --win_size 10 --train_ps 320
```

## Evaluation
The results reported in the paper are calculated by the `matlab` script used in [previous method](https://github.com/zhuyr97/AAAI2022_Unfolding_Network_Shadow_Removal/tree/master/codes). Details refer to `evaluation/measure_shadow.m`.
We also provide the `python` code for calculating the metrics in `test.py`, using `python test.py --cal_metrics` to print.

## Results
#### Evaluation on ISTD


#### Visual Results
<p align=center><img width="80%" src="doc/res.jpg"/></p>

Citation
-----
Preprint available [here](https://arxiv.org/pdf/2302.01650.pdf). 

```
nohup python train.py --warmup --win_size 8 --train_ps 256 > output.out 2>&1 &
```
```
python test.py --cal_metrics --win_size 8 
```
