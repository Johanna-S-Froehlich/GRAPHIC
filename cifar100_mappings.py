"""CIFAR-100 class metadata.

This module collects the CIFAR-100 fine labels, the official two-level
class hierarchy (20 superclasses, 100 fine labels) and an extra
"intuitive" grouping used in the paper.

It is imported by the metrics scripts to compute hierarchical metrics
such as assortativity. Keeping the static data here keeps the metrics
script focused on the analysis logic.
"""

from __future__ import annotations


# Fine labels in CIFAR-100 order (id = index in the list).
fine_labels = [
    'apple',              # id 0
    'aquarium_fish',      # id 1
    'baby',               # id 2
    'bear',               # id 3
    'beaver',             # id 4
    'bed',                # id 5
    'bee',                # id 6
    'beetle',             # id 7
    'bicycle',            # id 8
    'bottle',             # id 9
    'bowl',               # id 10
    'boy',                # id 11
    'bridge',             # id 12
    'bus',                # id 13
    'butterfly',          # id 14
    'camel',              # id 15
    'can',                # id 16
    'castle',             # id 17
    'caterpillar',        # id 18
    'cattle',             # id 19
    'chair',              # id 20
    'chimpanzee',         # id 21
    'clock',              # id 22
    'cloud',              # id 23
    'cockroach',          # id 24
    'couch',              # id 25
    'crab',               # id 26
    'crocodile',          # id 27
    'cup',                # id 28
    'dinosaur',           # id 29
    'dolphin',            # id 30
    'elephant',           # id 31
    'flatfish',           # id 32
    'forest',             # id 33
    'fox',                # id 34
    'girl',               # id 35
    'hamster',            # id 36
    'house',              # id 37
    'kangaroo',           # id 38
    'computer_keyboard',  # id 39
    'lamp',               # id 40
    'lawn_mower',         # id 41
    'leopard',            # id 42
    'lion',               # id 43
    'lizard',             # id 44
    'lobster',            # id 45
    'man',                # id 46
    'maple_tree',         # id 47
    'motorcycle',         # id 48
    'mountain',           # id 49
    'mouse',              # id 50
    'mushroom',           # id 51
    'oak_tree',           # id 52
    'orange',             # id 53
    'orchid',             # id 54
    'otter',              # id 55
    'palm_tree',          # id 56
    'pear',               # id 57
    'pickup_truck',       # id 58
    'pine_tree',          # id 59
    'plain',              # id 60
    'plate',              # id 61
    'poppy',              # id 62
    'porcupine',          # id 63
    'possum',             # id 64
    'rabbit',             # id 65
    'raccoon',            # id 66
    'ray',                # id 67
    'road',               # id 68
    'rocket',             # id 69
    'rose',               # id 70
    'sea',                # id 71
    'seal',               # id 72
    'shark',              # id 73
    'shrew',              # id 74
    'skunk',              # id 75
    'skyscraper',         # id 76
    'snail',              # id 77
    'snake',              # id 78
    'spider',             # id 79
    'squirrel',           # id 80
    'streetcar',          # id 81
    'sunflower',          # id 82
    'sweet_pepper',       # id 83
    'table',              # id 84
    'tank',               # id 85
    'telephone',          # id 86
    'television',         # id 87
    'tiger',              # id 88
    'tractor',            # id 89
    'train',              # id 90
    'trout',              # id 91
    'tulip',              # id 92
    'turtle',             # id 93
    'wardrobe',           # id 94
    'whale',              # id 95
    'willow_tree',        # id 96
    'wolf',               # id 97
    'woman',              # id 98
    'worm',               # id 99
]


# The official CIFAR-100 hierarchy: 20 superclasses, 5 fine labels each.
mapping_superclass_fine = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],                                     # 0
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],                                         # 1
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],                                           # 2
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],                                           # 3
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],                        # 4
    'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],       # 5
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],                                  # 6
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],                                  # 7
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],                                       # 8
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],                   # 9
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],                        # 10
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],            # 11
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],                             # 12
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],                             # 13
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],                                                      # 14
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],                                     # 15
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],                                   # 16
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],                           # 17
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],                                # 18
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],                                 # 19
}


# Coarsest grouping: which superclasses are "natural" vs "man-made".
mapping_make_superclass = {
    'natural': ['aquatic mammals', 'fish', 'flowers', 'fruit and vegetables', 'insects', 'large carnivores',
                'large natural outdoor scenes', 'large omnivores and herbivores', 'medium-sized mammals',
                'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees'],
    'man-made': ['food containers', 'household electrical device', 'household furniture',
                 'large man-made outdoor things', 'vehicles 1', 'vehicles 2'],
}


# Alternative grouping used in the paper, expressed directly over fine labels.
mapping_intuitive_fine = {
    'animal on land': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion',
                       'tiger', 'wolf', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox',
                       'porcupine', 'possum', 'raccoon', 'skunk', 'snail', 'spider', 'worm', 'dinosaur',
                       'lizard', 'snake', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'animal in water': ['crab', 'lobster', 'beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish',
                        'flatfish', 'ray', 'shark', 'trout', 'crocodile', 'turtle'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'home stuff': ['bottle', 'bowl', 'can', 'cup', 'plate', 'clock', 'computer_keyboard', 'lamp', 'telephone',
                   'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bicycle', 'bus', 'motorcycle',
                   'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'outside structures': ['cloud', 'forest', 'mountain', 'plain', 'sea', 'bridge', 'castle', 'house', 'road',
                           'skyscraper'],
    'plants': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'apple', 'mushroom', 'orange', 'pear',
               'sweet_pepper', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
}


def create_fine_labels_for_coarse_categories(mapping_superclass_fine, mapping_make_superclass):
    """Expand a make -> superclass map down to make -> fine labels.

    Used to produce the 'make_superclass' grouping (natural / man-made over
    fine labels) by composing the two existing mappings.

    Args:
        mapping_superclass_fine: superclass -> list of fine labels.
        mapping_make_superclass: coarse group -> list of superclasses.

    Returns:
        dict: coarse group -> flat list of fine labels.
    """
    fine_labels_for_coarse = {}
    for coarse_group, superclasses in mapping_make_superclass.items():
        fine_labels_for_coarse[coarse_group] = []
        for superclass in superclasses:
            if superclass in mapping_superclass_fine:
                fine_labels_for_coarse[coarse_group].extend(mapping_superclass_fine[superclass])
    return fine_labels_for_coarse


# Fine-label name -> CIFAR-100 index.
label_to_index = {label: idx for idx, label in enumerate(fine_labels)}


# Superclass communities expressed as lists of fine-label indices (one list
# per superclass, in declaration order).
superclass_communities = [
    [label_to_index[fine_label] for fine_label in fine_list]
    for fine_list in mapping_superclass_fine.values()
]


# Three named groupings of the 100 fine labels, ready to feed into
# the assortativity computation.
mappings = {
    "superclass_fine": mapping_superclass_fine,
    "intuitive_fine": mapping_intuitive_fine,
    "make_superclass": create_fine_labels_for_coarse_categories(
        mapping_superclass_fine, mapping_make_superclass
    ),
}
