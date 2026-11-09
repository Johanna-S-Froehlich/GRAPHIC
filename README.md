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

For training the model we call EffVit we create a second `venv` environment to avoid package conflicts. EffVit follows the training methodology from:

```
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

To train ResNet-50, you can use the `train.py` script in the top-level folder. For more information on the available options, run:
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


If you would like to train your model on Tiny ImageNet download the dataset by calling `python download_tiny_imagenet.py` from the top-level folder.

If you run your training on a cluster managed with SLURM, you can use the provided `submit_tiny_training.sh` script to submit your training job. Your email address for SLURM notifications in the `submit_tiny_training.sh` script has to be adjusted.

The script also contains the hyperparameters used in the paper, if you want to reproduce the results.

## How to create the confusion matrices

To create the confusion matrices for ResNet-50, use the `confusion_matrices_resnet.py` script. You specify the path to the model weights, the model architecture, and which layers to extract, plus an alpha value for balancing true and predicted labels. You can control which checkpoints to process via start/end indices and `--epoch_step`. For example:

```bash
source graphic/bin/activate
python confusion_matrices_resnet.py --checkpoint_dir $checkpoints_dir --output_dir $output_dir --model resnet50 --small_conv1 --only layer1 layer2 layer3 layer4 --start_index 1 --end_index 72 --epoch_step 5 --alpha 0.0 --iterations 1 --weight_decay
```

will create confusion matrices for `layer1` to `layer4` of ResNet-50, using every 5th checkpoint from the first to the 71st checkpoint, and training only on predicted labels (`alpha=0.0`). For the results from the paper we compute the confusion matrices layer-wise.

For more information on the available options, run:
```bash
source graphic/bin/activate
python confusion_matrices_resnet.py --help
```

This script can be adapted for other CNN-style architectures, but the model loading and layer extraction may need to be adjusted (the layer must output the relevant features cleanly).

For EffVit models, use the dedicated `confusion_matrices_vit.py` script. It pre-extracts and caches per-block class-token features, then trains a linear classifier on the cached features per block. Example call:

```bash
python confusion_matrices_vit.py --checkpoint_dir $checkpoints_dir --output_dir $output_dir --model efficient-vit --start_index 0 --end_index 10 --alpha 0.0 --num_classes 100 --depth 12 --weight_decay
```

Both scripts share their classifier-training and matrix-construction logic via `confusion_matrix_pipeline.py`. Each script writes two confusion matrices per checkpoint / layer-or-block: one on the held-out eval portion of the training set (`*_use_trainset.npy`) and one on the validation set (`*_use_valset.npy`).

SLURM scripts for both ResNet-50 and EffVit confusion matrix generation are provided in the `cluster_submit_confusion_matrix_job.sh` and `cluster_submit_vit_conf_matrix_job.sh` files, respectively. To create confusion matrices on Tiny ImageNet run `cluster_submit_vit_tiny_conf_matrix_job.sh`. As for the training scripts, adjust the email address.



## How to calculate the metrics and plot the graphs

To calculate the metrics and plot exemplary graphs use the `metrics.py` script. By default, the script will calculate all metrics used in the paper for each test set matrix, store them in a dictionary and save them as a json file. 

For more information about the script call:
```bash
source graphic/bin/activate
python metrics.py --help
```
Required arguments: `--matrix-dir` and `--output-dir`. The `--dataset` flag selects which class metadata to use (`cifar100` or `tiny-imagenet`, default `cifar100`); Tiny ImageNet has no class hierarchy, so the script automatically skips the assortativity computations. Optional flags: `--alpha` (default 1.0), `--train-set` (compute on the held-out train split rather than the test/validation split), `--plot-metrics` (line plots per layer over epochs), `--plot-graphs` (confusion-graph SVGs). `--seed` (default 42) sets the seed of the Louvain community detection.

Make sure to set the correct matrix and output directories. For ResNet-50 on CIFAR-100:

```bash
python metrics.py --matrix-dir resnet_conf_matrices --output-dir resnet_metrics --dataset cifar100 --alpha 0.0 --train-set --plot-metrics --plot-graphs
```

For EffVit on CIFAR-100:
```bash
python metrics.py --matrix-dir vit_conf_matrices --output-dir vit_metrics --dataset cifar100 --alpha 0.0 --train-set --plot-metrics --plot-graphs
```

For EffVit trained on Tiny ImageNet:
```bash
python metrics.py --matrix-dir vit_conf_matrices_tiny --output-dir vit_metrics_tiny --dataset tiny-imagenet --alpha 0.0 --train-set --plot-metrics --plot-graphs
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

## Citation

If you use this code, please cite:

```bibtex
@article{
fr{\"o}hlich2026the,
title={The Confusion is Real: {GRAPHIC} - A Network Science Approach to Confusion Matrices in Deep Learning},
author={Johanna S. Fr{\"o}hlich and Bastian Heinlein and Jan U. Claar and Hans Rosenberger and Vasileios Belagiannis and Ralf R. M{\"u}ller},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=UP9bx1WJwR},
note={J2C Certification}
}
```
