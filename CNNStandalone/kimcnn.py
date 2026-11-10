"""KimCNN-static (Kim 2014) with frozen pretrained or BERT-encoded inputs.

Layers are registered as named children so the GRAPHIC hooks (which call
``model.named_modules()``) can probe activations at any of:

    embedding   (B, L, E)                    -- Identity for BERT
    convs       (B, N*K, max_L')             -- concat of all branches, unpooled
    convs.0..   (B, K, L - k_i + 1)          -- per-branch unpooled features
    pool        (B, N*K)                     -- after global max-pool
    fc          (B, num_classes)

Reference: Y. Kim. "Convolutional Neural Networks for Sentence Classification."
EMNLP, 2014. https://arxiv.org/abs/1408.5882
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBranch(nn.Module):
    """One Conv2d + ReLU branch with kernel (k, embed_dim). Output is squeezed
    on the width dim so it has shape (B, K, L - k + 1)."""

    def __init__(self, kernel_size: int, kernel_num: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(1, kernel_num, (kernel_size, embed_dim))

    def forward(self, x):
        return F.relu(self.conv(x)).squeeze(3)


class _MultiConv(nn.Module):
    """Run N parallel _ConvBranch modules, right-pad each to equal length, and
    concat on the channel dim. Padding with zero is safe because all
    post-ReLU values are non-negative — padded positions never win the
    subsequent global max-pool."""

    def __init__(self, branches):
        super().__init__()
        for i, branch in enumerate(branches):
            self.add_module(str(i), branch)
        self._n = len(branches)

    def forward(self, x):
        maps = [self._modules[str(i)](x) for i in range(self._n)]
        max_len = max(m.size(2) for m in maps)
        padded = [F.pad(m, (0, max_len - m.size(2))) for m in maps]
        return torch.cat(padded, dim=1)


class _GlobalMaxPool(nn.Module):
    def forward(self, x):
        return F.max_pool1d(x, x.size(2)).squeeze(2)


class KimCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        kernel_sizes=(3, 4, 5),
        kernel_num: int = 100,
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=True, padding_idx=0
            )
            embed_dim = pretrained_embeddings.size(1)
        else:
            self.embedding = nn.Identity()
        self.embed_dim = embed_dim

        self.convs = _MultiConv(
            [_ConvBranch(k, kernel_num, embed_dim) for k in kernel_sizes]
        )
        self.pool = _GlobalMaxPool()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.convs(x)
        x = self.pool(x)
        x = self.dropout(x)
        return self.fc(x)
