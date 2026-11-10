# GRAPHIC - Graph-based Representation and Analysis of Predictions and Hidden-layer Interpretability via Confusion

This repository contains the code of two papers:

- **Image classification** (ResNet-50 and EffVit on CIFAR-100 and Tiny ImageNet): *The Confusion is Real: GRAPHIC - A Network Science Approach to Confusion Matrices in Deep Learning* (TMLR). The tag `v1.0.0-paper` is the exact code of that paper, and `v1.1.0` is the same pipeline with bug fixes and code cleanup, without the text-classification code.
- **Text classification** (KimCNN and Text GCN on R52 and WOS-46985): *Lost in Translation: A GRAPHIC Analysis of CNN and GNN Text Classifiers* (NLDL 2027). See [Text classification](#text-classification).

Both pipelines share `confusion_matrix_pipeline.py` and `metrics.py`. After the installation, the sections up to the metrics describe the image pipeline.


## Repository layout

| Path | Contents |
|---|---|
| `train.py`, `models/resnet.py`, `transforms/` | ResNet-50 training on CIFAR-100 |
| `VitStandalone/` | EffVit training, adapted from Irandoust et al. (with its own license, see `VitStandalone/LICENSE`) |
| `CNNStandalone/` | KimCNN training on R52 and WOS-46985 |
| `GCNStandalone/` | Text GCN: our patch for Yao's code, the data conversion and the TensorFlow requirements |
| `confusion_matrices_resnet.py`, `confusion_matrices_vit.py`, `confusion_matrices_cnn.py`, `confusion_matrices_gcn.py` | linear classifiers on the layer activations and the confusion matrices, one script per model |
| `confusion_matrix_pipeline.py`, `utils.py`, `models/LinearClassifier.py`, `models/CustomLossFunction.py`, `feature_extraction/`, `custom_datasets/` | shared code: classifier training, activation hooks, the loss with alpha, and helpers for training, model loading and reading the matrix files |
| `metrics.py` | graph metrics, community detection and plots, for all models |
| `cifar100_mappings.py`, `tinyimagenet_mappings.py`, `r52_mappings.py`, `wos46985_mappings.py` | class names and class groupings of the datasets |
| `download_*.py`, `text_split.py` | dataset downloads; the split of the text training sets for the linear classifiers |
| `cluster_submit_*.sh`, `cluster_run_*.sh` | SLURM scripts with the hyperparameters of the papers: the `submit` scripts submit the jobs, the `run` scripts are the jobs |
| `Full_Resolution_Graph_Plots/` | the graph plots of the TMLR paper in full resolution |

## Installation

The code runs on Linux with an NVIDIA GPU and Python 3.12. All package versions are pinned to the ones we tested. The image and text models need three environments, because EffVit and Text GCN (TensorFlow) conflict with the main one. On a cluster, load the Python module first (for example `module load python`). Then:

```bash
git clone https://github.com/Johanna-S-Froehlich/GRAPHIC.git
cd GRAPHIC

# ResNet-50, KimCNN, all confusion matrices and the metrics
python3.12 -m venv graphic
graphic/bin/pip install -r requirements.txt

# EffVit
python3.12 -m venv VitStandalone/train_vit
VitStandalone/train_vit/bin/pip install -r VitStandalone/requirements_vit.txt

# Text GCN: the TensorFlow environment, and Yao's code with our patch
python3.12 -m venv GCNStandalone/text_gcn
GCNStandalone/text_gcn/bin/pip install -r GCNStandalone/requirements.txt
git clone https://github.com/yao8839836/text_gcn GCNStandalone/text_gcn_tf2
(cd GCNStandalone/text_gcn_tf2 && git checkout 962223652e9bb164ac2d83cd09fc7b8845ce860b && git apply ../text_gcn_tf2.patch)
```

You only need the environments of the models you want to run. Yao's repository takes about 1.8 GB; `GCNStandalone/README.md` lists what our patch changes.

CIFAR-100 is downloaded automatically on the first run. For Tiny ImageNet, run `graphic/bin/python download_tiny_imagenet.py`. The text datasets and the GloVe vectors for KimCNN:

```bash
graphic/bin/python download_r52.py
graphic/bin/python download_wos46985.py

# GloVe: 2 GB download, 5.6 GB unpacked
glove_dir=${WORK:-$PWD}/glove_embeddings
mkdir -p $glove_dir
wget -P $glove_dir https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip $glove_dir/glove.840B.300d.zip -d $glove_dir && rm $glove_dir/glove.840B.300d.zip
```

The cluster scripts read GloVe from `$WORK/glove_embeddings/`, or from `glove_embeddings/` in this folder if `$WORK` is not set. If the Stanford server is slow, the same file is on Hugging Face: `https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip`.

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

You will find additional training instructions for EffVit in the repository VitStandalone in README.md. EffVit follows the training methodology from:

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

The linear classifiers are trained with Adam for at most 110 (ResNet-50) or 200 (EffVit) epochs, and the learning rate is reduced when the validation loss stops improving. Training stops early after 5 (ResNet-50) or 15 (EffVit) epochs without improvement, and the weights of the epoch with the lowest validation loss are used for the confusion matrices. This validation loss is computed on the validation set for ResNet-50 and on the held-out eval portion of the training set for EffVit.

SLURM scripts for both ResNet-50 and EffVit confusion matrix generation are provided in the `cluster_submit_confusion_matrix_job.sh` and `cluster_submit_vit_conf_matrix_job.sh` files, respectively. To create confusion matrices on Tiny ImageNet run `cluster_submit_vit_tiny_conf_matrix_job.sh`. As for the training scripts, adjust the email address.



## How to calculate the metrics and plot the graphs

To calculate the metrics and plot exemplary graphs use the `metrics.py` script. By default, the script will calculate all metrics used in the paper for each test set matrix, store them in a dictionary and save them as a json file. 

For more information about the script call:
```bash
source graphic/bin/activate
python metrics.py --help
```
Required arguments: `--matrix-dir` and `--output-dir`. The `--dataset` flag selects which class metadata to use (`cifar100`, `tiny-imagenet`, `r52` or `wos46985`, default `cifar100`); Tiny ImageNet has no class hierarchy, so the script automatically skips the assortativity computations. Optional flags: `--alpha` (default 1.0), `--train-set` (compute on the held-out train split rather than the test/validation split), `--plot-metrics` (line plots per layer over epochs), `--plot-graphs` (confusion-graph SVGs). `--seed` (default 42) sets the seed of the community detection. By default the communities are detected with Louvain; `--all-communities` adds Infomap and label propagation.

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

## Text classification

The text pipeline trains KimCNN (static, with GloVe or BERT embeddings) and Text GCN on R52 and WOS-46985, and then follows the same steps as above: linear classifiers on the layer activations, confusion matrices, metrics. Unlike the image pipeline, the linear classifiers train for a fixed number of epochs under a cosine schedule, without early stopping, and the held-out partition is a test set. The held-out confusion matrices are therefore named `*_use_testset.npy`.

The cluster scripts below contain the hyperparameters of the paper. Adjust the email address, the partition, the GPU type, the Python module and the proxy to your cluster. More details are in `CNNStandalone/README.md` and `GCNStandalone/README.md`.

The environments, Yao's code for Text GCN and the data are set up in [Installation](#installation). KimCNN with BERT encodes the documents on the first run and caches them in `r52_bert_cache/` and `wos46985_bert_cache/` (about 7 GB and 37 GB).

### KimCNN

```bash
bash cluster_submit_kimcnn.sh      # training: R52 and WOS-46985, GloVe and BERT
bash cluster_submit_kimcnn_cm.sh   # after training: confusion matrices for alpha 0 and 1, five iterations each
```

The confusion matrices are written to `cm_kimcnn_<dataset>_<embedding>/`. For each checkpoint they cover the layers `embedding`, `convs.0`, `convs.1`, `convs.2` and `pool`, and on the test set also the output layer `fc`, built from the predictions of the network itself.

### Text GCN

Text GCN uses Yao's implementation with our changes as a patch, set up in [Installation](#installation). `GCNStandalone/README.md` lists the changes.

```bash
bash cluster_submit_gcn.sh
```

One job per dataset prepares the data, trains the GCN and builds the confusion matrices for alpha 0 and 1. The activations are written to `$WORK/gcn_activations/` and the confusion matrices to `$WORK/gcn_cm/` (they use `$HOME` if `$WORK` is not set). They cover layer `1`, and on the test set also the output layer `2`.

### Metrics

```bash
source graphic/bin/activate
python metrics.py --matrix-dir cm_kimcnn_r52_glove --output-dir metrics_kimcnn_r52_glove --dataset r52 --alpha 0.0 --all-communities --plot-metrics
```

For R52 and WOS-46985 the script also computes the assortativity with the human grouping (`assort_human`) and the adjusted mutual information between the detected communities and the human grouping (`ami_<algorithm>_human`). The class names and human groupings are in `r52_mappings.py` and `wos46985_mappings.py`. The `accuracy` in the output is computed from the normalized confusion matrix. On these imbalanced datasets it is therefore the mean per-class recall in percent, not the share of correctly classified documents.

## Citation

If you use this code, please cite the paper for your pipeline. Image classification:

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

Text classification:

```bibtex
@inproceedings{
frohlich2026lost,
title={Lost in Translation: A {GRAPHIC} Analysis of {CNN} and {GNN} Text Classifiers},
author={Johanna S. Fr{\"o}hlich and Marcus Behr and Levi-Pascal V. G. Bohnacker and Bastian Heinlein and Ralf R. M{\"u}ller},
booktitle={Northern Lights Deep Learning Conference 2027},
year={2026},
url={https://openreview.net/forum?id=JrGKNyxGuV}
}
```
