# GRAPHIC - Graph-based Representation and Analysis of Predictions and Hidden-layer Interpretability via Confusion


## How to prepare the environment

We recommend working in a virtual environment to avoid conflicts with other packages. You can use `venv` or `conda` for this purpose.

For training Resnet-50 and creating the confusion matrices we suggest creating a `venv` environment and activate it as follows:
```bash
python -m venv graphic
source graphic/bin/activate
```

Next, install the package requirements by running:
```bash
pip install -r requirements.txt
deactivate
```

For training the model we call EffVit we create a second `venv` environment to avoid package conflicts according to:
```bash
@misc{irandoust2022training,
      title={{Training a Vision Transformer from scratch in less than 24 hours with 1 GPU}}, 
      author={Saghar Irandoust and Thibaut Durand and Yunduz Rakhmangulova and Wenjie Zi and Hossein Hajimirsadeghi},
      year={2022},
      eprint={2211.05187},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

The environment can be created using:
```bash
cd VitStandalone
python -m venv train_vit
source train_vit/bin/activate
```

Next, install the package requirements by running:
```bash
pip install -r requirements_vit.txt
deactivate
```



## How to train the models

To train ResNet-50, you can use the `train.py` script in the top Code_Graphic folder. For more information on the available options, run:
```bash
source graphic/bin/activate
python train.py --help
```

If you run your training on a cluster managed with SLURM, you can use the provided `cluster_submit_training_job.sh` script to submit your training job. Your email address for SLURM notifications in the `cluster_submit_training_job.sh` script has to be adjusted.
The script also contains the hyperparameters used in the paper, if you want to reproduce the results.

To train EffVit, you can use the `main.py` script in the VitStandalone folder with the environment created to train the visual transformer. 
```bash
source train_vit/bin/activate
```

You will find additional training instructions for EffVit in the repository VitStandalone in README.md.

If you run your training on a cluster managed with SLURM, you can use the provided `submit_cifar_training.sh` script to submit your training job. Your email address for SLURM notifications in the `submit_cifar_training.sh` script has to be adjusted.

The script also contains the hyperparameters used in the paper, if you want to reproduce the results.


If you would like to train your model on Tiny ImageNet download the dataset by calling `python download_tiny_imagenet.py` from the top Code_Graphic folder.

If you run your training on a cluster managed with SLURM, you can use the provided `submit_tiny_training.sh` script to submit your training job. Your email address for SLURM notifications in the `submit_tiny_training.sh` script has to be adjusted.

The script also contains the hyperparameters used in the paper, if you want to reproduce the results.

## How to create the confusion matrices

To create the confusion matrices for ResNet-50, you can use the `cifar-100-confusion-matrices.py` script. You will need to specify the path to the model weights, the model architecture, and which layers you want to extract, as well as an alpha value for balancing true and predicted labels. You can control for which checkpoints to run the script by specifying start and end indices corresponding to the natural sort order of the checkpoint files. For example:

```bash
source graphic/bin/activate
python cifar-100-confusion-matrices.py --checkpoint_dir $checkpoints_dir --output_dir $output_dir --model resnet50 --small_conv1 --only layer1 layer2 layer3 layer4 --start_index 1 --end_index 72 --epoch_step 5 --alpha 0.0 --iterations 1 --weight_decay
```

will create confusion matrices for `layer1` to `layer4` of ResNet-50, using every 5th checkpoint from the first to the 71st checkpoint, and training only on predicted labels (`alpha=0.0`). For the results from the paper we compute the confusion matrices layer-wise.

For more information on the available options, run:
```bash
source graphic/bin/activate
python cifar-100-confusion-matrices.py --help
```

This script can be adapted for other model architectures as well, but it would be necessary to adjust the model loading code accordingly. Extraction by layer name relies on the layer only outputting the relevant features, so model output may need to be modified to achieve this.

For this reason, creating confusion matrices for EffVit models is available in its own script `conf_matrices_vit.py`. You can run it in a similar way.

SLURM scripts for both ResNet-50 and EffVit confusion matrix generation are provided in the `cluster_submit_confusion_matrix_job.sh` and `cluster_submit_vit_conf_matrix_job.sh` files, respectively. To create confusion matrices on Tiny ImageNet run `cluster_submit_vit_tiny_conf_matrix_job.sh`. As for the training scripts, adjust the email address.



## How to calculate the metrics and plot the graphs

To calculate the metrics and plot exemplary graphs use the `metrics.py` script. By default, the script will calculate all metrics used in the paper for each test set matrix, store them in a dictionary and save them as a json file. 

For more information about the script call:
```bash
source graphic/bin/activate
python metrics.py --help
```
There are five options to consider. The parameter alpha has to be specified, the model chosen and three optional flags can be set: `--plot-metrics`, which will plot the metrics for each layer along the epochs, and save the plots in the output directory, `--train-set`, which leads to the computation for the trainset and `--plot-graphs`, which leads to the plotting of the confusion graphs.

Make sure to set the correct matrix and output directories. For ResNet-50 a possible call looks like the following:

```bash
python metrics.py --matrix-dir resnet_conf_matrices --output-dir resnet_metrics --alpha 0.0 --train-set --plot-metrics --plot-graphs

```

For EffVit a possible call is:
```bash
python metrics.py --matrix-dir vit_conf_matrices --output-dir vit_metrics --alpha 0.0 --train-set --plot-metrics --plot-graphs
```


For EffVit trained on Tiny ImageNet a possible call is:
```bash
python metrics_tiny.py --matrix-dir vit_conf_matrices_tiny --output-dir vit_metrics_tiny --alpha 0.0 --train-set --plot-metrics --plot-graphs
```


Note that the `file_matrix` function relies on the matrix files being named in the same way they are generated by our scripts, so be careful to use the same naming convention. 

If you want to do something else with these metrics, adapt this script accordingly or read the json somewhere else.

In the paper we use the external tool [Gephi](https://gephi.org/) to visualize the graphs, so keep in mind that your output will not look exactly the same as in the paper.

```bash
@paper{ICWSM09154,
author = {Mathieu Bastian and Sebastien Heymann and Mathieu Jacomy},
title = {Gephi: An Open Source Software for Exploring and Manipulating Networks},
conference = {International AAAI Conference on Weblogs and Social Media},
year = {2009},
}
```