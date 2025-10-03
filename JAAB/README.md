### JAAB: Multistage-Pipeline for Jotting and Assessing Actions in Boxing

## Overview

Boxing is a sport that consists of fast and explosive movement. In boxing there are a lot of actions happening sequentially and simultaneously in a short span of time. Our model is built for detecting those spontantenous action, specifically, whenever an athlete throws a punch we want to detect whether that punch land on the other athlete, miss the target, or blocked by the other athlete. Due to the niche of the task, we use our own custom dataset.

The project composes of two main folder. The folder labeled "JAABPortal" contains the website implementation for running the model as inference over the internet. The folder labeled "JAAB" contains all of the necessary component to develop and experiment with our model.

This folder contains the necessary component to run and reproduce the model


## Environment

The development and the experiment of the model is done within a Linux OS and utilizing Conda to maintain the environment. You can install the required packages for the project by following these steps
1. Build and activate a conda environment
2. With 'requirements.txt' in the folder specifying the versions of the various packages run the following command:
```
pip install -r requirements.txt
```


## Dataset
The dataset consist of two main compartment, the first one is the annotations and the second one is the necessary videos exported into individual frames. 

# Annotations
The annotations is located in the directory data/focus. The annotations consist of the event annotations and the track annotation. The event annotations represented by the files named 'train.json', 'test.json', 'val.json'. These files also represented the split of the dataset. The track annotation represented by file named 'track.json' which contains a dictionary of the tracked boxing glove from all of the videos in the dataset.

# Dataset frames
We provide the dataset by videos in .mp4 format as well as python script 'frames_as_jpg.py' to extract the videos onto individual frames. The python script can be executed using the following command,
```
python frames_as_jpg.py focus <path_to_the_video> -o <path_to_the_extracted_frames>
```


## Execution

The 'train_tdeed_exp.py' file is a modified file of "train_tdeed.py" from the original repository of T-DEED. The modification is for adjusting the train and evaluate algorithm to handle our custom dataset as well as any necessary changes adjusted for our model. You can execute the file using the following command:

```
python3 train_jaab.py --model Focus_small
```

You can control whether to train the whole model or just evaluate it using the 'only_test' parameter in the configuration file. If 'only_test' is set as true, make sure the checkpoint of the model exists in the 'checkpoints/Focus/Focus_small' directory

Before running the model, make sure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant config files. Same as T-DEED, run the script once with the 'mode' parameter set to 'store' to generate and save clip partitions. Afterward, you can set the 'mode' to 'load' to reuse the saved partitions for subsequent executions.


## Trained models

Model checkpoints can be found in the directory checkpoints/Focus/Focus_small with the name 'checkpoint_best.pt'


## Visualizing the output

After running the model on inference mode, the result of the prediction can be viewed as annotation in the 'save_data_dir/Focus_small' directory with the file named 'pred-test.json'. Using this json file the result can be visualized using visualizer.py script using the following command,
```
python visualizer.py
```
Before running the script make sure to correctly set up the paths variable such as 'video_folder', 'json_file', 'output_folder', 'track_dir'.


## Extra reminder
Make sure to set up the path correctly in files such as 'visualizer.py' and the config file in 'config/Focus/Focus_small.json' to make sure the model can run properly.


## References
Thanks to Xarles et. al we were able to accomplish our goal in this project
```
@inproceedings{xarles2024t,
  title={T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports Videos},
  author={Xarles, Artur and Escalera, Sergio and Moeslund, Thomas B and Clap{\'e}s, Albert},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3410--3419},
  year={2024}
}
```