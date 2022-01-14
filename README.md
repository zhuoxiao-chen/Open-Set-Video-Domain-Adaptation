


This is the official PyTorch implementation of our paper:
**Conditional Extreme Value Theory for Open Set Video Domain Adaptation**


Paper link: https://dl.acm.org/doi/10.1145/3469877.3490600

## Requirements
* Python 3.6, PyTorch 0.4, CUDA 10.0+
* The full requirements can be found in **requirements.txt**

## Datasets
Experiments are conducted on three datasets: UCF-HMDB<sub>small</sub>, UCF-HMDB<sub>full</sub>, UCF-Olympic.

The downloaded files need to store in `./dataset`.

Pre-extracted features and data lists can be downloaded as,
* Features
  * UCF: [download](https://www.dropbox.com/s/swfdjp7i79uddpf/ucf101-feat.zip?dl=0)
  * HMDB: [download](https://www.dropbox.com/s/c3b3v9zecen4dwo/hmdb51-feat.zip?dl=0)
  * Olympic: [training](https://www.dropbox.com/s/ynqw0yrnuqjmhhs/olympic_train-feat.zip?dl=0) | [validation](https://www.dropbox.com/s/mxl888ca06tg8wn/olympic_val-feat.zip?dl=0)
* Data lists
  * UCF-Olympic
    * UCF: [training list](https://www.dropbox.com/s/ennjl2g0m44srj4/list_ucf101_train_ucf_olympic-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/hz8wzj0bo7dhdx4/list_ucf101_val_ucf_olympic-feature.txt?dl=0)
    * Olympic: [training list](https://www.dropbox.com/s/cvoc2j7vw8r60lb/list_olympic_train_ucf_olympic-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/3jrnx7kxbpqnwau/list_olympic_val_ucf_olympic-feature.txt?dl=0)
  * UCF-HMDB<sub>small</sub>
    * UCF: [training list](https://www.dropbox.com/s/zss3383x90jkmvk/list_ucf101_train_hmdb_ucf_small-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/buslj4fb03olztu/list_ucf101_val_hmdb_ucf_small-feature.txt?dl=0)
    * HMDB: [training list](https://www.dropbox.com/s/exxejp3ppzkww94/list_hmdb51_train_hmdb_ucf_small-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/2b15gjehcisk8sn/list_hmdb51_val_hmdb_ucf_small-feature.txt?dl=0)
  * UCF-HMDB<sub>full</sub>
    * UCF: [training list](https://www.dropbox.com/s/8dq8xcekdi18a04/list_ucf101_train_hmdb_ucf-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/wnd6e0z3u36x50w/list_ucf101_val_hmdb_ucf-feature.txt?dl=0)
    * HMDB: [training list](https://www.dropbox.com/s/4bl7kt0er3mib19/list_hmdb51_train_hmdb_ucf-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/zdg3of6z370i22w/list_hmdb51_val_hmdb_ucf-feature.txt?dl=0)


## Datasets Split
For the open-set domain adaptation task, we need to keep source samples with **known** classes 0, 1, ..., C-1, C only and remove all source samples with classes C+1, C+2, ... We also need to change the unkown classes (C+1, C+2, ...) of target samples to (C+1), which is **unknown** class. To complete the datasets splitting, follow the steps below as an example of **Olympic → UCF**:

1. Rename the data list files ("org" means "original", which is used to backup the original list.):
   - "dataset/olympic/list_olympic_train_ucf_olympic-feature.txt" → "dataset/olympic/list_olympic_train_ucf_olympic-feature_org.txt" 
   - "dataset/ucf101/list_ucf101_train_ucf_olympic-feature.txt" → "dataset/ucf101/list_ucf101_train_ucf_olympic-feature_org.txt" 
   - "dataset/ucf101/list_ucf101_val_ucf_olympic-feature.txt" → "dataset/ucf101/list_ucf101_val_ucf_olympic-feature_org.txt"
2. In the script **open_set_data.py**, follow the comments to understand and set the variables as you need. 
3. Run **open_set_data.py**. You will get 3 processed list files.
4. According to the number of known classes you choose, remove lines of unknown classes in the file "data/classInd_ucf_olympic.txt". Also, remember to keep the original file. 

## Dataset Selection and Hyper-parameter Selection

In the main script **main.py**, there there are two blocks of code: **BLOCK 1** and **BLOCK 2**. These two blocks are used for select the dataset (put the names of 3 processed list files and modified class file) and hyper-parameters (best hyper-parameters are in paper), separately. 


```
########## BLOCK 1: Change Here for Different Datasets ##########
args.class_file = "data/classInd_ucf_olympic.txt"
args.train_source_list = "dataset/olympic/list_olympic_train_ucf_olympic-feature.txt"
args.train_target_list = "dataset/ucf101/list_ucf101_train_ucf_olympic-feature.txt"
args.val_list = "dataset/ucf101/list_ucf101_val_ucf_olympic-feature.txt"
########## END OF BLOCK 1 ##########
```

```
########## BLOCK 2: Change Here for Different Hyper-Parameters ##########
args.lambda_ = 0.214 # H->U 0.7 | U->O 0.19 | O->U 0.214
args.adv_param = 5 # H->U 0.10 | U->O 1.83 | O->U 5
args.EVT_threshold = 0.3 # H->U 0.45 | U->O  0.565 | O->U 0.3
########## END OF BLOCK 2 ##########
```

## Get Started

Once you have done the steps above, then you can run the code. 

```
python main.py
```

---
