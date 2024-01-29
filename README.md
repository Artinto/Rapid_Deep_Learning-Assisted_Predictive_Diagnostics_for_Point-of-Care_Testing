## Rapid Deep Learning-Assisted Predictive Diagnostics for Point-of-Care Testing
#### ✔ [Python 3.9](https://www.python.org/downloads/) and [Ubuntu 16.04](https://releases.ubuntu.com/16.04/) are required
#### ✔ [CUDA](https://developer.nvidia.com/cuda-10.2-download-archive)>=10.2 and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)>=8.0.2 are required.
#### ✔ [Anaconda](https://github.com/conda/conda) environment is recommended.
#### ✔ If you have any issues, please pull request.
## I. Enviroment setting
### 1. git install
```bash
$ sudo apt-get install git

$ git config --global user.name <user_name>
$ git config --global user.email <user_email>
```

<br>

### 2. Clone this Repository on your local path
```bash
$ cd <your_path>
$ git clone https://github.com/Artinto/TIMESAVER_Transforming_Point-of-care_Diagnostics
```

<br>

### 3. Create the virtual enviroment with conda (optional)
        
#### - Create your virtual enviroment
```bash
$ conda create -n <venv_name> python=3.9
```

#### - Activate your virtual environment
```bash
$ conda activate <venv_name>
```
**&rarr; Terminal will be...**   ```(venv_name) $ ```
  
#### -  Install requirements packages
```bash
(venv_name) $ pip install -r requirements.txt
```

<br>

## II. Download Dataset
### Dataset
- [Standard_sample.tar](https://drive.google.com/file/d/1eQWCfGTpKm9RF8PacSwa9JEib4hUCvx3/view?usp=sharing)
- [label_info.csv](https://drive.google.com/file/d/1nRP8ttMx2eDY74rdp_rHBpXxETM9sDW-/view?usp=sharing)

<br>

## III. File Structure
```
TIMESAVER_Transforming_Point-of-care_Diagnostics
├── README.md
├── requirements.txt
├── dataset
│   ├── __init__.py
│   └── dataset.py
├── models
│   ├── __init__.py
│   └── models.py
├── utils
│   ├── __init__.py
│   ├── log_util.py
│   ├── preprocess.py
│   ├── split_data.py
│   └── util.py
├── config.py
├── main.py
├── train.py
├── test.py
└── log
    └── train
        └── init_model
            ├── log.txt
            └── model_save
                ├── best_avg_model
                │   ├── best_density_model.pt
                │   └── best_target_model.pt
                └── best_avg_model.txt
```

```
Standard_sample
├── train
│   ├── 0
│   │   ├── sample_001
│   │   │   ├──  10.png
│   │   │   ├──  20.png
│   │   │   ├──  ...
│   │   │   └── 900.png
│   │   ├── sample_002
│   │   ├── ...
│   |   └── sample_214
│   ├── 200
│   ├── ...
│   └── 4096000
└── eval
    ├── 0
    ├── ...
    └── 4096000

```

<br>

## IV. Train Model
### Parameters Setting
```python3
# train.py

args = setting_params(
    mode='train',
    description='latent:1024+pretrain-r50+lstm',      
    data_path='./dataset/Standard_sample',
    label_info_path='dataset/label_info.csv',    
    use_cuda=True,                                  # GPU usage
    multi_gpu=False,                                # If you have two or more GPUs, it is recommended to set it to True
    num_epochs=500,
    train_batch_size=16,                            # Adjust according to GPU memory size
    eval_batch_size=2,                              # Adjust according to GPU memory size
    save_model=True,                                # True to save the best performing model (only available in train mode)
    use_frame=(0, 12, 1)                            # (start, end, step)=(0, 12, 1)=[0s, 10s, ... , 100s, 110s]
)
```

<br>

### Run
```bash
$ cd <your_path>/TIMESAVER_Transforming_Point-of-care_Diagnostics
$ python3 train.py
```

<br>

### Check the learning process with Tensorboard
#### Running Tensorboard
```bash
$ tensorboard --logdir="./log"
```
#### Connection
- The port number may change depending on the results of the above execution.
- http://localhost:6006/ 


<br>

## V. Test Model
### Parameters Setting

```python3
# test.py

args = setting_params(
    mode='test',
    description='latent:1024+pretrain-r50+lstm',      
    data_path='./dataset/Standard_sample',
    label_info_path='dataset/label_info.csv', 
    use_cuda=True, 
    multi_gpu=False,
    eval_batch_size=2,                              # Adjust according to GPU memory size
    load_saved_model=True,                          # Load a saved model
    path_saved_model='./model_save/best_avg_model', # Model weight path to load
    save_image=True,                                # Save the resulting image
    save_roc_curve=True                             # Save the ROC Curve
    use_frame=(0, 12, 1)                            # Use the same frame as training
)
```

<br>

### Run
```bash
$ python3 test.py
```
