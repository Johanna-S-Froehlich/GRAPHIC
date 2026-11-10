# KimCNN for the GRAPHIC text-classification pipeline

A standalone trainer for KimCNN ([Kim, 2014](https://arxiv.org/abs/1408.5882))
with frozen GloVe or pre-encoded BERT embeddings on R52 and WOS-46985.

This is the **CNN-static** variant: the embedding layer is non-trainable, only
the convolutional and classifier layers learn. Layer names match the GRAPHIC
hook convention so the confusion-matrix script at the repo root can train the
linear classifiers on the `embedding`, `convs.0`, `convs.1`, `convs.2` and `pool`
activations.

## Environment

KimCNN uses the `graphic` environment of the main README.

## Data

From the repo root:

```bash
python download_r52.py          # writes r52/
python download_wos46985.py     # writes wos46985/
```

The GloVe vectors have to be downloaded separately, see the main README.

## Train

From the repo root, for example with the hyperparameters of the paper for R52:

```bash
# GloVe
python CNNStandalone/train_kimcnn.py --dataset r52 --data-dir r52 \
    --embedding-type glove --glove-path ${WORK:-$PWD}/glove_embeddings/glove.840B.300d.txt \
    --max-len 512 --epochs 50 --batch-size 64 --lr 1e-3 --weight-decay 0 \
    --scheduler cosine --checkpoint-every 3 --output-dir kimcnn_r52_glove

# BERT (encodes the documents on the first run)
python CNNStandalone/train_kimcnn.py --dataset r52 --data-dir r52 \
    --embedding-type bert --bert-cache r52_bert_cache/bert_uncased_512.pt \
    --max-len 512 --epochs 50 --batch-size 64 --lr 3e-4 --weight-decay 1e-4 \
    --scheduler cosine --checkpoint-every 3 --output-dir kimcnn_r52_bert
```

`cluster_submit_kimcnn.sh` submits all four configurations. A checkpoint is saved
before training (`kimcnn_epoch_0.pt`), after the first epoch, every
`--checkpoint-every` epochs, and at the end (`kimcnn_final.pt`).

## Confusion matrices

```bash
python confusion_matrices_cnn.py --dataset r52 --data-dir r52 --embedding-type glove \
    --glove-path ${WORK:-$PWD}/glove_embeddings/glove.840B.300d.txt --max-len 512 \
    --checkpoint-dir kimcnn_r52_glove --output-dir cm_kimcnn_r52_glove --alpha 0.0 \
    --lc-lr 0.01 --weight-decay 1e-3 --lc-epochs 100 --save-confusions
```

`--save-confusions` also writes the misclassified test documents per class. For
WOS-46985 the paper uses `--lc-lr 0.003`. `cluster_submit_kimcnn_cm.sh` submits all
configurations.

## Training protocol

A modernized variant of Kim (2014): Adam instead of Adadelta, cosine annealing
over a fixed number of epochs without early stopping, and optional weight decay
instead of the l2-norm constraint. The final checkpoint is the trained model.
Dropout is 0.5, the batch size 64 and the maximum sequence length 512.

|               | R52 GloVe | R52 BERT | WoS GloVe | WoS BERT |
|---------------|-----------|----------|-----------|----------|
| Learning rate | 1e-3      | 3e-4     | 1e-3      | 1e-3     |
| Weight decay  | 0         | 1e-4     | 0         | 0        |
| Epochs        | 50        | 50       | 100       | 100      |

The dataset is split 80/20 per class into the network training set and the test
set. The network training set is split again per class into the set the linear
classifiers train on (80 %) and the set the training-side confusion matrices are
built on (20 %, see `text_split.py`). KimCNN and Text GCN use the same splits.

## Citation

```bibtex
@inproceedings{kim2014convolutional,
  title = {Convolutional Neural Networks for Sentence Classification},
  author = {Kim, Yoon},
  booktitle = {EMNLP},
  year = {2014}
}
```

For the GRAPHIC methodology, see the citations in the main README.
