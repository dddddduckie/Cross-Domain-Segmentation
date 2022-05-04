import argparse
from utils.file_op import load_config


def create_parser(mode='G2C'):
    if mode == 'G2C':
        config = load_config("./configs/gta2cityscapes.yaml")
    parser = argparse.ArgumentParser(description="Train models for cross-domain semantic segmentation")
    parser.add_argument("--source-data-dir", type=str, default=config["source_data_dir"],
                        help="path to the source domain dataset")
    parser.add_argument("--target-data-dir", type=str, default=config["target_data_dir"],
                        help="path to the target domain dataset")
    parser.add_argument("--source-list-dir", type=str, default=config["source_list_dir"],
                        help="path to the source domain data list")
    parser.add_argument("--target-list-dir", type=str, default=config["target_list_dir"],
                        help="path to the target domain data list")
    parser.add_argument("--log-dir", type=str, default=config["log_dir"],
                        help="path to save the log")
    parser.add_argument("--ckpt-dir", type=str, default=config["ckpt_dir"],
                        help="restore the checkpoint from last iteration")
    parser.add_argument("--result-dir", type=str, default=config["result_dir"],
                        help="path to save the segmentation map after visualization")
    parser.add_argument("--restore", type=bool, default=config["restore"],
                        help="restore from last iteration or train from scratch")
    parser.add_argument("--learning-rate", type=float, default=config["base_lr"],
                        help="initial learning rate for training")
    parser.add_argument("--maximum-lr", type=float, default=config["max_lr"],
                        help="maximum learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=config["weight_decay"],
                        help="weight decay for the optimizer")
    parser.add_argument("--step-size", type=int, default=config["step_size"],
                        help="step size for adjust learning rate")
    parser.add_argument("--save-iter", type=int, default=config["save_iter"],
                        help="save frequency of the model")
    parser.add_argument("--clip-gradient", type=bool, default=config["clip_gradient"],
                        help="clip the gradient during training")
    parser.add_argument("--batch-size", type=int, default=config["batch_size"],
                        help="batch size used for training")
    parser.add_argument("--num-workers", type=int, default=config["num_workers"],
                        help="number of data loading workers")
    parser.add_argument("--tensorboard", type=bool, default=config["use_tensorboard"],
                        help="use tensorboard to store scalar information")
    parser.add_argument("--max-norm", type=int, default=config["max_norm"],
                        help="max norm for gradient clip")
    parser.add_argument("--max-iter", type=int, default=config["max_iter"],
                        help="maximum iteration during training")
    parser.add_argument("--num-classes", type=int, default=config["num_classes"],
                        help="number of classes in the dataset")
    parser.add_argument("--ignore-label", type=int, default=config["ignore_label"],
                        help="label to ignore when computing the loss")
    parser.add_argument("--momentum", type=int, default=config["momentum"],
                        help="momentum used for SGD training")

    return parser.parse_args()
