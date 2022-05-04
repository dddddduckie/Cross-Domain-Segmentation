import torch
import os
import json
import numpy as np
import torch.utils.data as data
from model.model_factory import create_model
from parser import create_parser
from torch.backends import cudnn
from dataset.dataloader import create_dataset
from utils.evaluation_metric import confusion_matrix, mIoU, per_cls_iou
from utils.data_visualization import visualize_segmap
from utils.file_op import mkdir
from PIL import Image

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_default_tensor_type(torch.FloatTensor)

TORCH_VERSION = torch.__version__
TORCH_CUDA_VERSION = torch.version.cuda
CUDNN_VERSION = str(cudnn.version())
DEVICE_NAME = torch.cuda.get_device_name()

# cudnn.benchmark = True
cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = create_parser()
ckpt_name = 'Cross_Domain_Segmentation_G2C_45000.pth.tar'


def evaluation(model, dataloader):
    # result dir
    result_dir = parser.result_dir
    # label dir
    label_file = os.path.join(parser.target_list_dir, 'info.json')

    # label mapping
    with open(label_file, 'r') as fp:
        info = json.load(fp)
    label_map = {}
    label = info["label"]
    for i in range(len(label)):
        label_map[i] = label[i]

    cf_matrix = np.zeros((parser.num_classes,) * 2)
    num_iters = len(dataloader)

    # evaluation mode
    model.eval()
    model.to(device)
    for i, samples in enumerate(dataloader):
        images, labels, name = samples
        images = images.to(device)
        labels = labels.cpu().numpy()
        with torch.no_grad():
            _, output = model(images)
        output = output.data.cpu().numpy()
        preds = np.argmax(output, axis=1)

        # color map only supports batch size=1
        color_map = visualize_segmap(preds[0], dataset="cityscapes") * 255.0
        # color_image = Image.fromarray(color_map.astype(np.uint8))
        cf_matrix += confusion_matrix(preds, labels, num_classes=parser.num_classes)

        if (i + 1) % 100 == 0:
            print("iteration: {}/{}".format(i + 1, num_iters))

        name = name[0].split('/')[-1]
        # color_image.save('%s/%s' % (parser.result_dir, name))

    cur_mIoU = round(mIoU(cf_matrix) * 100, 2)
    per_class_iou = per_cls_iou(cf_matrix)
    for i in range(len(per_class_iou)):
        print('===>' + label_map[i] + ':\t' + str(round(per_class_iou[i] * 100, 2)))

    return cur_mIoU


def main():
    # checkpoint dir
    ckpt_dir = parser.ckpt_dir
    mkdir(parser.result_dir)

    # dataset preparation
    _, _, test_data = create_dataset(mode='G2C')
    test_data_loader = data.DataLoader(test_data, batch_size=1, shuffle=False,
                                       num_workers=parser.num_workers, pin_memory=True)

    # create model and optimizer
    model = create_model(num_classes=parser.num_classes, name='DeepLab')

    if parser.restore:
        print("loading checkpoint...")
        checkpoint = torch.load(os.path.join(ckpt_dir, ckpt_name))
        model.load_state_dict(checkpoint['model'])

    print("start evaluating...")
    print("pytorch version: " + TORCH_VERSION + ", cuda version: " + TORCH_CUDA_VERSION +
          ", cudnn version: " + CUDNN_VERSION)
    print("available graphical device: " + DEVICE_NAME)
    os.system("nvidia-smi")

    mIoU = evaluation(model, test_data_loader)

    print("finished evaluating, the mIoU on the evaluation set is: " + str(mIoU))


if __name__ == '__main__':
    main()
