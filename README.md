# IDD-Lite-Unet

# Requirement 

* Pytorch=0.4
* Visdom
* Opencv
* matplotlib
* pydensecrf

To directly create the conda virtual environemt please use the requirements.yml 


# Training procedure 

1. Download the dataset in a folder "dataset" and arrange the data in the following structure: Combine the training and validation dataset.

```bash
├── dataset
│   ├── images
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   ├── annotation
│   │   ├── train
│   │   ├── val
```

2. Start the visdom server by the command 
~~~bash
python -m visdom.server
~~~
2. Execute the command 
```bash
bash run.sh
```
# Weights
[https://drive.google.com/file/d/1yxCbStft75gTOriWQLGtqS_5l_OAvYnR/view?usp=sharing]

# Testing procedure

1. For single scale testing:
```bash
python test.py
```
2. For multi-scale testing:
```bash
python multiscale_testing.py
```
3. For single scale testing with CRF post processing 
```bash
python test_with_postprocessing.py
```

# Visualization 

Left side : predicted classes  :  Right side : Ground Truth 

![Visualization during training](images/visual.png?raw=true "sample of generated data")

# Acknowledgement
One hundred layer Tiramisu [https://github.com/bfortuner/pytorch_tiramisu]

