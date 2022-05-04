import numpy as np
from utils.file_op import load_config


def get_palette(dataset):
    """
    get the palette according to the dataset
    """
    support_dataset = ['voc', 'cityscapes', 'kitti', 'bdd100k']
    assert support_dataset.__contains__(dataset), "unsupported dataset"

    config = load_config("/media/sdb/duckie/Cross-Domain-Segmentation/configs/palette.yaml")
    if dataset == "voc":
        palette = config["voc"]
    elif dataset == "cityscapes":
        palette = config["cityscapes"]

    return np.array(palette)


def visualize_segmap(seg_map, dataset):
    assert type(seg_map) == np.ndarray, \
        "segmentation map must be converted to numpy array"

    palette = get_palette(dataset)
    height = seg_map.shape[0]
    width = seg_map.shape[1]
    color_map = np.zeros(shape=(height, width, 3))
    for i in range(height):
        for j in range(width):
            if seg_map[i][j] < 19:
                color_map[i][j] = palette[seg_map[i][j].astype(np.int)]
    color_map = color_map / 255.0

    return color_map

