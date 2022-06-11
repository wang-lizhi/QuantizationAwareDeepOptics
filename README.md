# QDO

This repository provides the code for the paper *Quantization-aware Deep Optics for Diffractive Snapshot Hyperspectral Imaging* (CVPR 2022).

## Environment

Firstly, use Anaconda to create a virtual Python 3.8 environment with necessary dependencies from the **environment.yaml** file in the code.

```
conda env create -f ./environment.yaml
```

Then, activate the created environment and continue to train or test.

## Train

### Dataset Preparation

To train the QDO model for hyperspectral imaging, the dataset should be downloaded to your computer in advance.
(e.g., [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/), [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/), or [Harvard](http://vision.seas.harvard.edu/hyperspec/index.html).)


Then, edit the ```DATASET_PATH``` dictionary in **util/data/dataset_loader.py** to indicate the name and path to your dataset. Here is an example:
```
DATASET_PATH = {
  "dataset_name1": "/PATH1/TO/YOUR/DATSET1",
  "dataset_name2": "/PATH2/TO/YOUR/DATSET2",
  "dataset_name3": "/PATH2/TO/YOUR/DATSET3"
}
```
And there should be three directories in your dataset path: [train, validation, test] to indicate which part should be used for training, validation, and testing.

### Argument Configuration

After the dataset is prepared, configure the ```dataset_name``` and ```dataset_loader_func_name``` of ```controlled_training_args``` dictionary in **tasks/hyperspectral.py**.

The ```dataset_loader_func_name``` can be any function provided in **util/data/dataset_loader.py** or any function you implement using TensorFlow Dataset. 
(You should also put your customized dataset loader function in *util/data/dataset_loader.py* and set the ```dataset_loader_func_name``` to your customized function name. So that the model can automatically import and use it.)

The python file in tasks could be duplicated and renamed to store different task configurations, including dataset, training options, loss, network arguments, etc.

Current **tasks/hyperspectral.py** has already provided an example configuration for training using the ICVL dataset.


### Start Training

After the configuration, the training can be started with the following commands:
```bash
python main.py --task hyperspectral \
               --doe_material SK1300 --quantization_level 4 \
               --alpha_blending --adaptive_quantization \
               --sensor_distance_mm 50 --scene_depth_m 1 \
               --tag QDOA4LevelHyperspctralTraining \
```
This example shows a 4-level QDO+A model using SK1300 as the DOE material. Its sensor distance is 50mm, and the scene depth is 1m.

 ```--task``` argument is the name of the python file in the **tasks** package (without ".py").

 ```--doe_material``` argument indicates the material refractive index used for DOE simulation. Supported options: SK1300, BK7, NOA61.

 ```--quantization_level``` argument indicates the level count for the DOE quantization.

 ```--alpha_blending``` argument (optional) indicates whether to use alpha-blending for quantization-aware training. If this option is not given, the model will use STE.

 ```--adaptive_quantization``` argument (optional) defines whether to use adaptive quantization (QDO+A model) or not (QDO model).

 ```--sensor_distance_mm``` argument defines the distance between the DOE to the sensor plane in millimeter.

 ```--scene_depth_m``` argument defines the distance between the scene to the DOE in meter.

 ```--tag``` argument is a label that makes it easier to manage checkpoints and log files.

When the training starts, the trainer will save checkpoints and current task arguments into **./checkpoint/** as 2 JSON files named **controlled_model_args.json** and **controlled_training_args.json**. These files are important for the trainer to continue training and necessary for the evaluator to test the model. Visualization summary results, including DOE height maps, PSFs, and encoded images, will also be saved to the *./logs/* directory. Tensorboard can be used for viewing these results.

The checkpoint of a QDO+A 4-level model trained on ICVL is provided [here](https://mega.nz/file/FkNkWBpA#H3PXA0DIuDVl3G2xcTeVPp5Yx6lc7A03tyoqNaRGL8k). You can put extracted files into **./checkpoint/<TAG_NAME>** for later test. The **<TAG_NAME>** can be any value you want as the directory name.

## Test

After training, evaluation can be performed using the following commands:
```bash
# For QDO models, using the test set
python evaluator.py --checkpoint_dir CHEKPOINT_DIR  \
                    --tag_name TAG_NAME \
                    --tag_vars TAG_VARS
                    
# For conventional DO model, using the test set
python evaluator.py --checkpoint_dir CHEKPOINT_DIR \
                    --tag_name TAG_NAME \
                    --tag_vars TAG_VARS \
                    --test_q TEST_QUANTIZATION_LEVEL
                    
# For QDO models, using the real RGB data
python evaluator.py --checkpoint_dir CHEKPOINT_DIR \
                    --tag_name TAG_NAME \
                    --tag_vars TAG_VARS \
                    --real_data_dir REAL_DATA_DIR
```

```--checkpoint_dir``` argument is the name of the sub directory in **./checkpoint/**.

```--tag_name``` argument is the inner tag name indicating the sub-directory in ```-checkpoint_dir``` given above.

```--tag_vars``` argument (optional) is the string value to insert in the %s placeholder of tag_name. Do not set this argument if there is no "%s" in your ```--tag_name``` argument.

```--test_q``` argument (optional) indicates the quantization level used during the test. Only the conventional DO model test needs this argument.

```--real_data_dir``` argument (optional) is the path to the directory storing real captured RGB images (PNG files).

The evaluator will output test results into **./eval-res/**, including .csv files with test metrics, visualized RGB images, and corresponding hyperspectral .mat files.

# Citation
If our code is useful in your reseach work, please consider citing our paper.
```
@inproceedings{li2022quantization,
  title={Quantization-aware Deep Optics for Diffractive Snapshot Hyperspectral Imaging},
  author={Li, Lingen and Wang, Lizhi and Song, Weitao and Zhang, Lei and Xiong, Zhiwei and Huang, Hua},
  booktitle={CVPR},
  year={2022},
  pages={19780-19789}
}
```
