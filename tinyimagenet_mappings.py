"""Tiny ImageNet class metadata.

Tiny ImageNet has 200 fine-grained classes and no explicit class hierarchy
in this project, so this module only exposes the fine-label list and the
derived ``label_to_index`` map. The metrics script skips assortativity
when there is no hierarchy to compute over.
"""

from __future__ import annotations


# Fine labels in Tiny ImageNet order (id = index in the list).
fine_labels = [
    'goldfish',                  # id 0
    'European fire salamander',  # id 1
    'bullfrog',                  # id 2
    'tailed frog',               # id 3
    'American alligator',        # id 4
    'boa constrictor',           # id 5
    'trilobite',                 # id 6
    'scorpion',                  # id 7
    'black widow',               # id 8
    'tarantula',                 # id 9
    'centipede',                 # id 10
    'goose',                     # id 11
    'koala',                     # id 12
    'jellyfish',                 # id 13
    'brain coral',               # id 14
    'snail',                     # id 15
    'slug',                      # id 16
    'sea slug',                  # id 17
    'American lobster',          # id 18
    'spiny lobster',             # id 19
    'black stork',               # id 20
    'king penguin',              # id 21
    'albatross',                 # id 22
    'dugong',                    # id 23
    'Chihuahua',                 # id 24
    'Yorkshire terrier',         # id 25
    'golden retriever',          # id 26
    'Labrador retriever',        # id 27
    'German shepherd',           # id 28
    'standard poodle',           # id 29
    'tabby',                     # id 30
    'Persian cat',               # id 31
    'Egyptian cat',              # id 32
    'cougar',                    # id 33
    'lion',                      # id 34
    'brown bear',                # id 35
    'ladybug',                   # id 36
    'fly',                       # id 37
    'bee',                       # id 38
    'grasshopper',               # id 39
    'walking stick',             # id 40
    'cockroach',                 # id 41
    'mantis',                    # id 42
    'dragonfly',                 # id 43
    'monarch',                   # id 44
    'sulphur butterfly',         # id 45
    'sea cucumber',              # id 46
    'guinea pig',                # id 47
    'hog',                       # id 48
    'ox',                        # id 49
    'bison',                     # id 50
    'bighorn',                   # id 51
    'gazelle',                   # id 52
    'Arabian camel',             # id 53
    'orangutan',                 # id 54
    'chimpanzee',                # id 55
    'baboon',                    # id 56
    'African elephant',          # id 57
    'lesser panda',              # id 58
    'abacus',                    # id 59
    'academic gown',             # id 60
    'altar',                     # id 61
    'apron',                     # id 62
    'backpack',                  # id 63
    'bannister',                 # id 64
    'barbershop',                # id 65
    'barn',                      # id 66
    'barrel',                    # id 67
    'basketball',                # id 68
    'bathtub',                   # id 69
    'beach wagon',               # id 70
    'beacon',                    # id 71
    'beaker',                    # id 72
    'beer bottle',               # id 73
    'bikini',                    # id 74
    'binoculars',                # id 75
    'birdhouse',                 # id 76
    'bow tie',                   # id 77
    'brass',                     # id 78
    'broom',                     # id 79
    'bucket',                    # id 80
    'bullet train',              # id 81
    'butcher shop',              # id 82
    'candle',                    # id 83
    'cannon',                    # id 84
    'cardigan',                  # id 85
    'cash machine',              # id 86
    'CD player',                 # id 87
    'chain',                     # id 88
    'chest',                     # id 89
    'Christmas stocking',        # id 90
    'cliff dwelling',            # id 91
    'computer keyboard',         # id 92
    'confectionery',             # id 93
    'convertible',               # id 94
    'crane',                     # id 95
    'dam',                       # id 96
    'desk',                      # id 97
    'dining table',              # id 98
    'drumstick',                 # id 99
    'dumbbell',                  # id 100
    'flagpole',                  # id 101
    'fountain',                  # id 102
    'freight car',               # id 103
    'frying pan',                # id 104
    'fur coat',                  # id 105
    'gasmask',                   # id 106
    'go-kart',                   # id 107
    'gondola',                   # id 108
    'hourglass',                 # id 109
    'iPod',                      # id 110
    'jinrikisha',                # id 111
    'kimono',                    # id 112
    'lampshade',                 # id 113
    'lawn mower',                # id 114
    'lifeboat',                  # id 115
    'limousine',                 # id 116
    'magnetic compass',          # id 117
    'maypole',                   # id 118
    'military uniform',          # id 119
    'miniskirt',                 # id 120
    'moving van',                # id 121
    'nail',                      # id 122
    'neck brace',                # id 123
    'obelisk',                   # id 124
    'oboe',                      # id 125
    'organ',                     # id 126
    'parking meter',             # id 127
    'pay-phone',                 # id 128
    'picket fence',              # id 129
    'pill bottle',               # id 130
    'plunger',                   # id 131
    'pole',                      # id 132
    'police van',                # id 133
    'poncho',                    # id 134
    'pop bottle',                # id 135
    'potters wheel',             # id 136
    'projectile',                # id 137
    'punching bag',              # id 138
    'reel',                      # id 139
    'refrigerator',              # id 140
    'remote control',            # id 141
    'rocking chair',             # id 142
    'rugby ball',                # id 143
    'sandal',                    # id 144
    'school bus',                # id 145
    'scoreboard',                # id 146
    'sewing machine',            # id 147
    'snorkel',                   # id 148
    'sock',                      # id 149
    'sombrero',                  # id 150
    'space heater',              # id 151
    'spider web',                # id 152
    'sports car',                # id 153
    'steel arch bridge',         # id 154
    'stopwatch',                 # id 155
    'sunglasses',                # id 156
    'suspension bridge',         # id 157
    'swimming trunks',           # id 158
    'syringe',                   # id 159
    'teapot',                    # id 160
    'teddy',                     # id 161
    'thatch',                    # id 162
    'torch',                     # id 163
    'tractor',                   # id 164
    'triumphal arch',            # id 165
    'trolleybus',                # id 166
    'turnstile',                 # id 167
    'umbrella',                  # id 168
    'vestment',                  # id 169
    'viaduct',                   # id 170
    'volleyball',                # id 171
    'water jug',                 # id 172
    'water tower',               # id 173
    'wok',                       # id 174
    'wooden spoon',              # id 175
    'comic book',                # id 176
    'plate',                     # id 177
    'guacamole',                 # id 178
    'ice cream',                 # id 179
    'ice lolly',                 # id 180
    'pretzel',                   # id 181
    'mashed potato',             # id 182
    'cauliflower',               # id 183
    'bell pepper',               # id 184
    'mushroom',                  # id 185
    'orange',                    # id 186
    'lemon',                     # id 187
    'banana',                    # id 188
    'pomegranate',               # id 189
    'meat loaf',                 # id 190
    'pizza',                     # id 191
    'potpie',                    # id 192
    'espresso',                  # id 193
    'alp',                       # id 194
    'cliff',                     # id 195
    'coral reef',                # id 196
    'lakeside',                  # id 197
    'seashore',                  # id 198
    'acorn',                     # id 199
]


# Fine-label name -> Tiny ImageNet index.
label_to_index = {label: idx for idx, label in enumerate(fine_labels)}
