import tensorflow as tf

DATASET_PATH = "data"

IMAGE_SIZE = [512, 512]
PATH = DATASET_PATH + "/tfrecords-jpeg-512x512"
AUTO = tf.data.experimental.AUTOTUNE

TRAINING_FILENAMES = tf.io.gfile.glob(PATH + "/train/*.tfrec")
VALIDATION_FILENAMES = tf.io.gfile.glob(PATH + "/val/*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob(PATH + "/test/*.tfrec")

CLASSES = [
    "pink primrose",  # 0
    "hard-leaved pocket orchid",  # 1
    "canterbury bells",  # 2
    "sweet pea",  # 3
    "wild geranium",  # 4
    "tiger lily",  # 5
    "moon orchid",  # 6
    "bird of paradise",  # 7
    "monkshood",  # 8
    "globe thistle",  # 9
    "snapdragon",  # 10
    "colt's foot",  # 11
    "king protea",  # 12
    "spear thistle",  # 13
    "yellow iris",  # 14
    "globe-flower",  # 15
    "purple coneflower",  # 16
    "peruvian lily",  # 17
    "balloon flower",  # 18
    "giant white arum lily",  # 19
    "fire lily",  # 20
    "pincushion flower",  # 21
    "fritillary",  # 22
    "red ginger",  # 23
    "grape hyacinth",  # 24
    "corn poppy",  # 25
    "prince of wales feathers",  # 26
    "stemless gentian",  # 27
    "artichoke",  # 28
    "sweet william",  # 29
    "carnation",  # 30
    "garden phlox",  # 31
    "love in the mist",  # 32
    "cosmos",  # 33
    "alpine sea holly",  # 34
    "ruby-lipped cattleya",  # 35
    "cape flower",  # 36
    "great masterwort",  # 37
    "siam tulip",  # 38
    "lenten rose",  # 39
    "barberton daisy",  # 40
    "daffodil",  # 41
    "sword lily",  # 42
    "poinsettia",  # 43
    "bolero deep blue",  # 44
    "wallflower",  # 45
    "marigold",  # 46
    "buttercup",  # 47
    "daisy",  # 48
    "common dandelion",  # 49
    "petunia",  # 50
    "wild pansy",  # 51
    "primula",  # 52
    "sunflower",  # 53
    "lilac hibiscus",  # 54
    "bishop of llandaff",  # 55
    "gaura",  # 56
    "geranium",  # 57
    "orange dahlia",  # 58
    "pink-yellow dahlia",  # 59
    "cautleya spicata",  # 60
    "japanese anemone",  # 61
    "black-eyed susan",  # 62
    "silverbush",  # 63
    "californian poppy",  # 64
    "osteospermum",  # 65
    "spring crocus",  # 66
    "iris",  # 67
    "windflower",  # 68
    "tree poppy",  # 69
    "gazania",  # 70
    "azalea",  # 71
    "water lily",  # 72
    "rose",  # 73
    "thorn apple",  # 74
    "morning glory",  # 75
    "passion flower",  # 76
    "lotus",  # 77
    "toad lily",  # 78
    "anthurium",  # 79
    "frangipani",  # 80
    "clematis",  # 81
    "hibiscus",  # 82
    "columbine",  # 83
    "desert-rose",  # 84
    "tree mallow",  # 85
    "magnolia",  # 86
    "cyclamen ",  # 87
    "watercress",  # 88
    "canna lily",  # 89
    "hippeastrum ",  # 90
    "bee balm",  # 91
    "pink quill",  # 92
    "foxglove",  # 93
    "bougainvillea",  # 94
    "camellia",  # 95
    "mallow",  # 96
    "mexican petunia",  # 97
    "bromelia",  # 98
    "blanket flower",  # 99
    "trumpet creeper",  # 100
    "blackberry lily",  # 101
    "common tulip",  # 102
    "wild rose",  # 103
]