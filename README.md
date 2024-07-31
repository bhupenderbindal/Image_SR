# ImgSR
==============================

Image super-resolution methods.

Project Organization 
------------
    |-- image_basics
    |   |-- data
    |   |   |-- raw                      <- The original, immutable data dump.
    |   |   |-- interim
    |   |   |   |-- .gitkeep
    |   |   |-- processed
    |   |   |-- readme.md
    |   |-- LICENSE
    |   |-- results          <- Results are save here
    |   |   |-- srcnn
    |   |   |-- kernelgan
    |   |-- environment30June.yml
    |   |-- src                <- Source code for use in this project.
    |   |   |-- data           <- Scripts to download or generate data
    |   |   |   |-- split_dataset_train_val.py
    |   |   |   |-- .gitkeep
    |   |   |   |-- make_dataset.py
    |   |   |   |-- split_dataset_train_val_test.py
    |   |   |   |-- imgsplitter.py
    |   |   |   |-- paired_data_generator_withalignment_samplewise.py
    |   |   |   |-- __init__.py
    |   |   |   |-- pair_cropping.py
    |   |   |   |-- align_images.py
    |   |   |-- visualization
    |   |   |   |-- visualize.py
    |   |   |   |-- .gitkeep
    |   |   |   |-- plot_utils.py
    |   |   |   |-- metrics_paradox.py
    |   |   |   |-- __init__.py
    |   |   |-- srcnn                    <- Scripts for example-based models
    |   |   |   |-- LICENSE
    |   |   |   |-- solver_abc.py
    |   |   |   |-- progress_bar.py
    |   |   |   |-- main.py
    |   |   |   |-- smooth_tiled_predictions.py
    |   |   |   |-- prediction_using_smooth_blending.py
    |   |   |   |-- .gitignore
    |   |   |   |-- super_resolve_rgb.py
    |   |   |   |-- __init__.py
    |   |   |   |-- inference.py                  <- Scripts for inference
    |   |   |   |-- README.md
    |   |   |-- my_logger.py
    |   |   |-- models
    |   |   |   |-- .gitkeep
    |   |   |   |-- opencv_sr.py
    |   |   |   |-- __init__.py
    |   |   |-- utils
    |   |   |   |-- patch_and_combine.py
    |   |   |   |-- utils.py
    |   |   |-- losses
    |   |   |   |-- FDL.py
    |   |   |   |-- models.py
    |   |   |   |-- contextual_los
    |   |   |   |-- __init__.py
    |   |   |   |-- lpipss
    |   |   |   |   |-- pretrained_networks.py
    |   |   |   |   |-- __init__.py
    |   |   |   |   |-- trainer.py
    |   |   |   |   |-- lpips.py
    |   |   |-- __init__.py
    |   |   |-- KernelGAN-master_v1                   <- Scripts for SISR
    |   |   |   |-- train.py
    |   |   |   |-- util.py
    |   |   |   |-- loss.py
    |   |   |   |-- results
    |   |   |   |-- LICENSE.txt
    |   |   |   |-- configs.py
    |   |   |   |-- kernelganv1.sh
    |   |   |   |-- KernelGAN.yml
    |   |   |   |-- learner.py
    |   |   |   |-- .gitignore
    |   |   |   |-- networks.py
    |   |   |   |-- imresize.py
    |   |   |   |-- kernelGAN.py
    |   |   |   |-- noise_estimation.py
    |   |   |   |-- data.py
    |   |   |   |-- pytorch_ZSSR_master
    |   |   |   |   |-- LICENSE
    |   |   |   |   |-- configs.py
    |   |   |   |   |-- utils.py
    |   |   |   |   |-- imresize.py
    |   |   |   |   |-- __init__.py
    |   |   |   |   |-- run_ZSSR_single_input.py
    |   |   |   |   |-- simplenet.py
    |   |   |   |   |-- ZSSR.py
    |   |   |   |   |-- run_ZSSR.py
    |   |   |   |   |-- README.md
    |   |   |   |-- README.md
    |   |   |-- __init__.pyc
    |   |-- test_environment.py
    |   |-- docs 
    |   |-- .gitignore
    |   |-- setup.py
    |   |-- logs
    |   |-- .env
    |   |-- directory_tree.py
    |   |-- Makefile                <- Makefile with commands like `make data` or `make train`
    |   |-- directory-tree.md
    |   |-- requirements_30June.txt 
    |   |-- README.md               <- The top-level README for developers using this project.



3. Getting Started:

    - Create the conda environment and install the required packages.

    - Make commands in the Makefile can be used to generate data, train and evaluate models.
    
    - Data Preparation: Place raw data in the data/raw/ directory, organised into subfolders for each dataset. 

            data/raw/all_data/Images_set1
            data/raw/all_data/Images_set2

            The preprocessing scripts process the data and save paired images in data/processed.



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
