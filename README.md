# oxsfg-steel
A repo for localisation and feature extraction of steel facilities in remote sensing imagery.

5any@wa2021

## Installation

## Developer Install

To setup as a developer, clone this repo and then:

```
pip install -e .[dev]
pre-commit install
```

## Useage

### Build a dataset


#### Authorization

Building a dataset requires authorization from Earth Engine and Google Maps API (to use the basemap).
To use the Earth Engine REST API, you must be an [Earth Engine user](https://earthengine.google.com/new_signup/), you must [create a GCP service account](https://support.google.com/a/answer/7378726?hl=en), download the credentials json, and [register the service account email with the Earth Engine REST API](https://signup.earthengine.google.com/#!/service_accounts).
To use the Google Maps API, the [Static Maps API](https://console.cloud.google.com/google/maps-apis/start?_ga=2.23728058.1203202661.1642099908-149915178.1642099908) must be enabled, and the key must be downloaded to a text file.
The paths to the json credentials and textfile API key must be specified in the YAML config file.

#### Builder CLI

To build a dataset, use the cli utilities in `bin/make_dataset.py`.
A dataset for Sentinel-2 and Google Basemap imagery can be made using the example YAML provided in [conf/]
Call the script like so:

    python bin/make_dataset.py make-pointcentre-dataset ./data/oxsfg_steel.gpkg ./conf/build_multi_dataset.yaml ./data/test_ds/
    
### Run ML Experiments

This repository uses [Sacred](https://sacred.readthedocs.io/en/stable/experiment.html) to configure and run ML experiments. 
Experiment default configurations can be found and set in [conf/] and artefacts can be found in [experiments/sacred/].
The path to the experiment file can be set in [oxsfg/steel/experiment.py] or can be called via the command line like:

    python runner.py with <path/to/your/conf.yaml>


[Tensorboard](https://www.tensorflow.org/tensorboard) logging is also enabled. 
Run a tensorboard server using:

    tensorboard --logdir experiments/tensorboard
    
#### Panel Data Experiment e.g. Steel Plant Capacity

Default configuration can be found at [conf/main.yaml].
Parameters are:

    dataset_parameters:
        gdf_path: <path/to/your/gdf.gpkg>       # path to your geodataframe dataset
        data_root: <path/to/your/data/>         # filepath where imagery data can be found
        target_column: capacity                 # column to regress on
        log_target: true                        # log-norm the target variable
        log_norm: 3.0                           # divide the log-normed target by a scalar
        data_keys: [BASEMAP]                    # which data variable to use
        is_categorical: false                   # train on a categorical variable
        zscore_path: <path/to/your/zscore.pkl>  # z-score your training input data
        pretrain_norm: true                     # normalise by torchvision IMAGENET params 
        crop:                                   
            BASEMAP: [100,100,100,100]          # crop image data
        resize_dim:
            BASEMAP: [300,300]                  # resize to [m,n] pixels

    training_parameters:                        # training parameters self-explanatory
        batch_size: 8
        val_split: 0.1
        test_split: 0.1
        num_workers: 8
        max_epochs: 8
        gpus: 1

    model_parameters:                           
        channels_last: true                     # training data has [Batch, H, W, Channels] dimension
        
        
#### Sequence Data Experiment e.g. Steel Plant Age

Default configuration can be found at [conf/main_sequence.yaml].
The parameters are:

    dataset_parameters:
        gdf_path: <path/to/your/gdf.gpkg>       # path to your geodataframe dataset
        data_root: <path/to/your/data/>         # filepath where imagery data can be found
        target_column: year                     # age column to define 'activation' year
        pretrain_norm: true                     # normalise by torchvision IMAGENET params 
        sequence_len: 150                       # number of revisits to use in sequence
        resize_dim: [256,256]                   # resize each revisit to [m,n] px

    training_parameters:                        # training params self explantory
        batch_size: 4
        val_split: 0.1
        test_split: 0.1
        num_workers: 6
        max_epochs: 40
        gpus: 1

    model_parameters:                           
        input_size: 256                         # input size in pixels for pretrained encoder
        hidden_size: 12                         # hidden fully-connected layer size
