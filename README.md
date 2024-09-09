## Step 0) ⚠️ Warning !!
If you want to execute this project, there will some problems, this repo isn't complete currently.
If there is any questions, please check here: https://github.com/s3prl/s3prl/tree/main/s3prl/pretrain

## Step 1) Prepare data
1) Download the LibriSpeech raw data from [here](http://www.openslr.org/12).
    - These sets are used for pretraining:
        - train-clean-100 [6.3G]
        - train-clean-360 [23G]
        - train-other-500 [30G]
    - The LibriSpeech directory after download and unzip should look like this: 
      ![](https://i.imgur.com/PdAOXjq.png)
2) **(IMPORTANT)** Generate the meta data directory `len_for_bucket/` for bucketing to accelerate training: 
```bash
python3 preprocess/generate_len_for_bucket.py -i PATH_TO_YOUR/LibriSpeech/
```

## Step 2) Modifiy runner config
1) Open `S3PRL/pretrain/tera/config_runner.yaml`:
    - This is the default runner config of tera, and it will be taken by default if not specified otherwise.
    - To assign another config other then default, you can use the `-c` argument, for example:
      `-u tera -c another_new_config_runner.yaml`
      
2) Change the paths in `config_runner.yaml`:
    - Change the following to your own path:
    ```yaml
    libri_root: '/media/andi611/1TBSSD/LibriSpeech/'
    file_path: 'data/len_for_bucket' 
    ```
3) Other training settings to care about:
    - Check these attributes in `config_runner.yaml`:
    ```yaml
    n_epochs: 100
    total_steps: -1
    gradient_accumulate_steps: 8
    train_batch_size: 32
    sets: ['train-clean-100', 'train-clean-360', 'train-other-500']
    ```
    - If `n_epochs` is given, `total_steps` will be ignored.
    - Set `n_epochs` to `-1` if you want to use `total_steps` instead
    - The acutal batch size = `gradient_accumulate_steps` * `train_batch_size`
    - Modify the `sets` list to choose pretraining subsets.

    
## Step 3) Start training
### Distiller
- Command:
```bash
python run_pretrain.py -u multi_distiller -g pretrain/multi_distiller/config_model.yaml -n multi_distill_model_test
```

### If using exciting checkpoint
```bash
python run_pretrain.py -u multi_distiller -n multi_distill_model_test -e "path to check point"

```
