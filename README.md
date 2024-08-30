# ImgSR
==============================

Image super-resolution methods.

Project Organization (higher-level)
------------
    |-- image_basics
    |   |-- data
    |   |   |-- raw                      <- The original image data.
    |   |   |-- interim
    |   |   |   |-- .gitkeep
    |   |   |-- processed                <- Image-pairs generated from raw data.
    |   |   |-- readme.md
    |   |-- LICENSE
    |   |-- results                     <- Results are save here
    |   |   |-- srcnn
    |   |   |-- kernelgan
    |   |-- environment30June.yml
    |   |-- src                                 <- Source code for use in this project.
    |   |   |-- data                            <- Scripts to download or generate data
    |   |   |   |-- split_dataset_train_val.py
    |   |   |   |-- make_dataset.py
    |   |   |   |-- split_dataset_train_val_test.py
    |   |   |   |-- imgsplitter.py
    |   |   |   |-- paired_data_generator_withalignment_samplewise.py
    |   |   |   |-- pair_cropping.py
    |   |   |   |-- align_images.py
    |   |   |-- visualization
    |   |   |   |-- visualize.py
    |   |   |   |-- plot_utils.py
    |   |   |   |-- metrics_paradox.py
    |   |   |-- srcnn                                 <- Scripts for example-based models
    |   |   |   |-- LICENSE
    |   |   |   |-- solver_abc.py                     <- Parent class for example based model training
    |   |   |   |-- progress_bar.py
    |   |   |   |-- main.py                           <- Scripts for setting up the training and testing the example-based model
    |   |   |   |-- smooth_tiled_predictions.py
    |   |   |   |-- prediction_using_smooth_blending.py
    |   |   |   |-- super_resolve_rgb.py
    |   |   |   |-- inference.py                      <- Script for inference
    |   |   |   |-- README.md
    |   |   |-- my_logger.py
    |   |   |-- models
    |   |   |   |-- .gitkeep
    |   |   |   |-- opencv_sr.py
    |   |   |-- utils
    |   |   |   |-- patch_and_combine.py
    |   |   |   |-- utils.py
    |   |   |-- losses
    |   |   |   |-- FDL.py              
    |   |   |   |-- models.py
    |   |   |   |-- contextual_los
    |   |   |   |-- lpipss
    |   |   |   |   |-- pretrained_networks.py
    |   |   |   |   |-- trainer.py
    |   |   |   |   |-- lpips.py
    |   |   |-- KernelGAN-master_v1                   <- Scripts for SISR
    |   |   |   |-- train.py
    |   |-- test_environment.py
    |   |-- docs 
    |   |-- .gitignore
    |   |-- setup.py
    |   |-- logs
    |   |-- .env
    |   |-- directory_tree.py
    |   |-- Makefile                <- Makefile with commands like `make SETNUM=3 create_samplewise_dataset` etc.
    |   |-- directory-tree.md
    |   |-- requirements_30June.txt 
    |   |-- README.md               <- The top-level README for this project.



3. Getting Started:

    - Create the conda environment and install the required packages.

    - Make commands in the Makefile can be used to generate data, train and evaluate models. 
        Make commands are used just to shorten the long cli command, one can use the full commands.
    
    - Data Preparation: Place raw data in the data/raw/ directory, organised into subfolders for each dataset. 

            data/raw/all_data/Images_set1
            data/raw/all_data/Images_set2
            data/raw/all_data/Images_set3 (only test data samples)

            NOTE: 
            - paths were hardcoded into scripts for convienience.
            - since, the test data consist of new samples. The data splitting works by splitting only the train into train and val. 
            Test data can be stored in Image_set3.
            
            The preprocessing scripts process the data and save paired images in data/processed.

            ``` 
            make SETNUM=3 create_samplewise_dataset 
            ``` 

    - Training
        - Example-based

            srcnn: 
            ```
            make EXP=experimentname EP=1600 DT=fdata CT=y BS=16 SETNUM=2 LOSSTYPE=l2 srcnn 
            ```

            dbpn:
            ``` 
            make EXP=experimentname EP=1600 DT=fdata CT=rgb BS=16 SETNUM=2 LOSSTYPE=cobi dbpn
            ```
        - single-image based (kernelgan+zssr)
            ```
            make kernelganset2
            ```
    - Testing script are executed together in /src/srcnn/main.
        They can be executed separately with the trained model setting the test directory in the script and model directory in the cli.
        - on paired images
        ```
        python -m src.srcnn.super_resolve_rgb  --save_dir ./model --channeltype rgb
        ```
        - on image with unknown ground truth
        ```
        python -m src.srcnn.inference  --save_dir ./model --channeltype rgb
        ```
        NOTE: patch size super resolution can be increased or decreased based on the memory of the system.





--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
