# Text-GCN for the GRAPHIC text-classification pipeline

We use Yao's reference implementation of Text-GCN
([Yao et al., 2019](https://arxiv.org/abs/1809.05679)),
[yao8839836/text_gcn](https://github.com/yao8839836/text_gcn). It has no license,
so this repository does not include it. `text_gcn_tf2.patch` contains our changes
to it instead.

## Setup

From the repo root:

```bash
git clone https://github.com/yao8839836/text_gcn GCNStandalone/text_gcn_tf2
cd GCNStandalone/text_gcn_tf2
git checkout 962223652e9bb164ac2d83cd09fc7b8845ce860b
git apply ../text_gcn_tf2.patch
cd ../..
```

Yao's repository takes about 1.8 GB, because it includes the datasets of Yao et al.
`cluster_run_gcn.sh` stops if the folder is missing or the patch is not applied.

The patch changes the following:

- `train.py`, `layers.py`, `models.py`, `inits.py`, `metrics.py`: ported to
  TensorFlow 2 (eager execution disabled, `tf.compat.v1` shims), and the GPU is
  used instead of the CPU.
- `train.py`: a fixed seed, and a fixed number of epochs under cosine annealing
  instead of early stopping. The activations of both graph-convolution layers are
  written to `epoch_N_{train,test}_activations_{1,2}.txt`, every 5 epochs plus the
  first and last epoch.
- `layers.py`: the activations are taken after the ReLU, as for KimCNN.
  The sparse matrix product is split, so the large WOS-46985 adjacency fits the
  GPU kernel.
- `models.py`: the learning rate is an input of the model, for the cosine schedule,
  and `--num_layers` adds graph-convolution layers (the paper uses 2).
- `build_graph.py`: no validation split, so the GCN trains on all training
  documents, like KimCNN. The classes are sorted, so the output columns of the GCN
  follow the class order of the confusion matrices. Fixed seeds.
- `build_graph.py`, `remove_words.py`, `train.py`: WOS-46985 is added to the
  supported datasets.
- `utils.py`: two lines updated for current NumPy and SciPy.

`prepare_for_gcn.py` converts our R52 and WOS-46985 data into Yao's file layout,
with the same splits as KimCNN.

## Environment

Text GCN needs a separate venv from the `graphic` one (TF2 + CUDA vs PyTorch + CUDA
conflicts). `requirements.txt` pins the versions we tested, for Python 3.12:

```bash
cd GCNStandalone
python -m venv text_gcn
source text_gcn/bin/activate
pip install -r requirements.txt
```

`prepare_for_gcn.py` and the confusion matrices run in the `graphic` environment.

## Data preparation

From the repo root:

```bash
python download_r52.py          # writes r52/
python download_wos46985.py     # writes wos46985/
```

Then convert into Yao's format:

```bash
python GCNStandalone/prepare_for_gcn.py --dataset R52
python GCNStandalone/prepare_for_gcn.py --dataset WOS46985
```

This populates `GCNStandalone/text_gcn_tf2/data/<NAME>.txt` (metadata) and
`text_gcn_tf2/data/corpus/<NAME>.txt` (raw text).

## Train

```bash
cd GCNStandalone/text_gcn_tf2
python remove_words.py R52
PYTHONHASHSEED=0 python build_graph.py R52
python train.py        R52 --activations_dir activations --hidden1 200 --learning_rate 0.02
```

For WOS-46985 the paper uses `--hidden1 536 --learning_rate 0.08`, with 150 epochs,
dropout 0.5 and no weight decay on both datasets. `PYTHONHASHSEED` makes the graph
reproducible between runs.

`confusion_matrices_gcn.py` at the repo root reads the activations. The linear
classifiers use `--lc-lr 0.3` on R52 and `--lc-lr 0.01` on WOS-46985, with
`--weight-decay 0 --lc-epochs 100`.

`cluster_run_gcn.sh` runs all of these steps for one dataset and `cluster_submit_gcn.sh`
submits both datasets. The script also sets what the environment needs:
`LD_LIBRARY_PATH` for the CUDA libraries in the venv, without which TensorFlow falls
back to the CPU, and `NLTK_ALLOW_PROXIED_URLOPEN=1` for downloading the stopwords
through a proxy.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yao2019graph,
  title     = {Graph Convolutional Networks for Text Classification},
  author    = {Yao, Liang and Mao, Chengsheng and Luo, Yuan},
  booktitle = {AAAI},
  year      = {2019},
  pages     = {7370--7377}
}
```

For R52: Cardoso-Cachopo, "Improving Methods for Single-label Text
Categorization", PhD Thesis, 2007.

For WOS-46985: Kowsari et al., "HDLTex: Hierarchical Deep Learning for Text
Classification", ICMLA 2017.

The README of Yao's repository describes the original TensorFlow 1 setup; with
the patch, use the environment above instead.
