import torch
import os
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
from model.model_factory import create_model
from parser import create_parser
from torch.backends import cudnn
from dataset.dataloader import create_dataset
from utils.net_util import adjust_learning_rate, create_optimizer
from utils.loss import compute_loss
from utils.file_op import add_summary, sava_checkpoint
from utils.evaluation_metric import confusion_matrix, mIoU
from model.discriminator import Discriminator

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_default_tensor_type(torch.FloatTensor)

TORCH_VERSION = torch.__version__
TORCH_CUDA_VERSION = torch.version.cuda
CUDNN_VERSION = str(cudnn.version())
DEVICE_NAME = torch.cuda.get_device_name()

cudnn.benchmark = True
cudnn.enabled = True
# cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = create_parser()
mode = 'G2C'

# hyper parameters
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET_1 = 0.0002
LAMBDA_ADV_TARGET_2 = 0.001
LEARNING_RATE_D = 1e-4
SOURCE_LABEL = 0
TARGET_LABEL = 1
BETAS = (0.9, 0.99)


def entropy_map(pred):
    """
    convert predictions to entropy map
    """
    b, c, h, w = pred.size()
    return -torch.mul(pred, torch.log2(pred + 1e-30)) / np.log2(c)


def validation(model, dataloader):
    cf_matrix = np.zeros((parser.num_classes,) * 2)
    num_iters = len(dataloader)

    # evaluation mode
    model.eval()
    print("now validating...")
    for i, samples in enumerate(dataloader):
        images, labels, _ = samples
        images = images.to(device)
        labels = labels.cpu().numpy()
        with torch.no_grad():
            _, output = model(images)
        output = output.data.cpu().numpy()
        preds = np.argmax(output, axis=1)
        cf_matrix += confusion_matrix(preds, labels, num_classes=parser.num_classes)

        if (i + 1) % 100 == 0:
            print("iteration: {}/{}".format(i + 1, num_iters))

    cur_mIoU = round(mIoU(cf_matrix) * 100, 2)
    print("current mIoU on the validation set: " + str(cur_mIoU))

    return cur_mIoU


def train(model, discriminator, optimizer, source_data_iter,
          target_data_iter, val_data, start_iter, last_mIoU):
    # record the validation result
    best_mIoU = last_mIoU
    best_iter = start_iter
    D1 = discriminator['D1']
    D2 = discriminator['D2']

    model.train()
    model.to(device)
    D1.train()
    D1.to(device)
    D2.train()
    D2.to(device)

    optimizer_G = optimizer['G']
    optimizer_D1 = optimizer['D1']
    optimizer_D2 = optimizer['D2']

    for i in range(start_iter, parser.max_iter + 1):
        loss_seg1_value = 0
        loss_seg2_value = 0
        loss_D1_value = 0
        loss_D2_value = 0
        loss_adv1_target_value = 0
        loss_adv2_target_value = 0

        optimizer_G.zero_grad()
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate(optimizer=optimizer_G, cur_iter=i, ini_lr=parser.learning_rate,
                             step_size=parser.step_size, max_iter=parser.max_iter, mode='poly')
        adjust_learning_rate(optimizer=optimizer_D1, cur_iter=i, ini_lr=LEARNING_RATE_D,
                             step_size=parser.step_size, max_iter=parser.max_iter, mode='poly')
        adjust_learning_rate(optimizer=optimizer_D2, cur_iter=i, ini_lr=LEARNING_RATE_D,
                             step_size=parser.step_size, max_iter=parser.max_iter, mode='poly')

        # train G
        # froze gradient in discriminator
        for params in D1.parameters():
            params.requires_grad = False
        for params in D2.parameters():
            params.requires_grad = False

        _, batch = source_data_iter.__next__()
        images, labels = batch
        images = images.to(device)
        labels = labels.long().to(device)

        # train with source data
        output1_source, output2_source = model(images)
        loss_seg1 = compute_loss(output1_source, labels, name='ce', ignore_index=parser.ignore_label)
        loss_seg2 = compute_loss(output2_source, labels, name='ce', ignore_index=parser.ignore_label)
        loss_seg = loss_seg2 + LAMBDA_SEG * loss_seg1
        loss_seg.backward()
        loss_seg1_value += loss_seg1.item()
        loss_seg2_value += loss_seg2.item()

        # train with target data
        _, batch = target_data_iter.__next__()
        images, _, _ = batch
        images = images.to(device)
        output1_target, output2_target = model(images)
        output_map1 = D1(entropy_map(F.softmax(output1_target)))
        output_map2 = D2(entropy_map(F.softmax(output2_target)))
        label_map1 = torch.FloatTensor(output_map1.data.size()).fill_(SOURCE_LABEL).to(device)
        label_map2 = torch.FloatTensor(output_map2.data.size()).fill_(SOURCE_LABEL).to(device)
        loss_adv1_target = compute_loss(output_map1, label_map1, name='bce')
        loss_adv1_target_value += loss_adv1_target.item()
        loss_adv2_target = compute_loss(output_map2, label_map2, name='bce')
        loss_adv2_target_value += loss_adv2_target.item()

        loss_adv_target = LAMBDA_ADV_TARGET_1 * loss_adv1_target + LAMBDA_ADV_TARGET_2 * loss_adv2_target
        loss_adv_target.backward()

        # train D
        # bring back gradient
        for params in D1.parameters():
            params.requires_grad = True
        for params in D2.parameters():
            params.requires_grad = True

        # train with source
        output1_source = output1_source.detach()
        output2_source = output2_source.detach()
        output_map1 = D1(entropy_map(F.softmax(output1_source)))
        output_map2 = D2(entropy_map(F.softmax(output2_source)))
        label_map1 = torch.FloatTensor(output_map1.data.size()).fill_(SOURCE_LABEL).to(device)
        label_map2 = torch.FloatTensor(output_map2.data.size()).fill_(SOURCE_LABEL).to(device)
        loss_D1 = compute_loss(output_map1, label_map1, name='bce')
        loss_D1 = loss_D1 / 2
        loss_D1.backward()
        loss_D1_value += loss_D1.item()
        loss_D2 = compute_loss(output_map2, label_map2, name='bce')
        loss_D2 = loss_D2 / 2
        loss_D2.backward()
        loss_D2_value += loss_D2.item()

        # train with target
        output1_target = output1_target.detach()
        output2_target = output2_target.detach()
        output_map1 = D1(entropy_map(F.softmax(output1_target)))
        output_map2 = D2(entropy_map(F.softmax(output2_target)))
        label_map1 = torch.FloatTensor(output_map1.data.size()).fill_(TARGET_LABEL).to(device)
        label_map2 = torch.FloatTensor(output_map2.data.size()).fill_(TARGET_LABEL).to(device)
        loss_D1 = compute_loss(output_map1, label_map1, name='bce')
        loss_D1 = loss_D1 / 2
        loss_D1.backward()
        loss_D1_value += loss_D1.item()
        loss_D2 = compute_loss(output_map2, label_map2, name='bce')
        loss_D2 = loss_D2 / 2
        loss_D2.backward()
        loss_D2_value += loss_D2.item()

        # clip gradient
        if parser.clip_gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=parser.max_norm)
            torch.nn.utils.clip_grad_norm_(D1.parameters(), max_norm=parser.max_norm)
            torch.nn.utils.clip_grad_norm_(D2.parameters(), max_norm=parser.max_norm)

        optimizer_G.step()
        optimizer_D1.step()
        optimizer_D2.step()

        print(
            "iteration: {}/{}, loss_seg1_value: {}, loss_seg2_value: {}, loss_D1_value: {},loss_D2_value: {}, "
            "loss_adv1_target_value: {},loss_adv2_target_value: {}"
                .format(i + 1, parser.max_iter, loss_seg1_value, loss_seg2_value, loss_D1_value, loss_D2_value,
                        loss_adv1_target_value, loss_adv2_target_value))

        if parser.tensorboard:
            saved_scalar = {
                'loss_seg1': loss_seg1_value,
                'loss_seg2': loss_seg2_value,
                'loss_D1': loss_D1_value,
                'loss_D2': loss_D2_value,
                'loss_adv1_target': loss_adv1_target_value,
                'loss_adv2_target': loss_adv2_target_value
            }
            add_summary(saved_scalar, i, parser.log_dir)

        if i % parser.save_iter == 0:
            cur_mIoU = validation(model, val_data)
            if cur_mIoU > best_mIoU:
                optimizer = {'G': optimizer_G.state_dict(),
                             'D1': optimizer_D1.state_dict(),
                             'D2': optimizer_D2.state_dict()}
                best_iter = i
                best_mIoU = cur_mIoU
                state_dict = {
                    'iter': best_iter,
                    'model': model.state_dict(),
                    'optimizer': optimizer,
                    'best_mIoU': best_mIoU
                }
                prefix = "Cross_Domain_Segmentation_" + mode
                sava_checkpoint(state_dict, parser.ckpt_dir, prefix=prefix)

    return best_mIoU, best_iter


def main():
    # dataset preparation
    source_data, target_data, val_data = create_dataset(mode='G2C')
    source_dataloader = Data.DataLoader(source_data, batch_size=parser.batch_size, shuffle=True,
                                        num_workers=parser.num_workers, pin_memory=True)
    target_dataloader = Data.DataLoader(target_data, batch_size=parser.batch_size, shuffle=True,
                                        num_workers=parser.num_workers, pin_memory=True)
    val_dataloader = Data.DataLoader(val_data, batch_size=parser.batch_size, shuffle=False,
                                     num_workers=parser.num_workers, pin_memory=True)
    source_dataloader_iter = enumerate(source_dataloader)
    target_dataloader_iter = enumerate(target_dataloader)

    save_dir = parser.ckpt_dir

    # create model and optimizer
    model = create_model(num_classes=parser.num_classes, name='DeepLab')
    D1 = Discriminator(num_classes=parser.num_classes)
    D2 = Discriminator(num_classes=parser.num_classes)

    optimizer_G = create_optimizer(model.get_optim_params(parser), lr=parser.learning_rate,
                                   momentum=parser.momentum, weight_decay=parser.weight_decay, name="SGD")
    optimizer_D1 = create_optimizer(D1.parameters(), lr=LEARNING_RATE_D, name="Adam", betas=BETAS)
    optimizer_D2 = create_optimizer(D2.parameters(), lr=LEARNING_RATE_D, name="Adam", betas=BETAS)

    optimizer_G.zero_grad()
    optimizer_D1.zero_grad()
    optimizer_D2.zero_grad()

    start_iter = 1
    last_mIoU = 0

    if parser.restore:
        print("loading checkpoint...")
        checkpoint = torch.load(save_dir)
        start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['model'])
        optimizer_G.load_state_dict(checkpoint['optimizer']['G'])
        optimizer_D1.load_state_dict(checkpoint['optimizer']['D1'])
        optimizer_D2.load_state_dict(checkpoint['optimizer']['D2'])
        last_mIoU = checkpoint['best_mIoU']

    print("start training...")
    print("pytorch version: " + TORCH_VERSION + ", cuda version: " + TORCH_CUDA_VERSION +
          ", cudnn version: " + CUDNN_VERSION)
    print("available graphical device: " + DEVICE_NAME)
    os.system("nvidia-smi")

    discriminator = {'D1': D1, 'D2': D2}
    optimizer = {'G': optimizer_G, 'D1': optimizer_D1, 'D2': optimizer_D2}

    best_mIoU, best_iter = train(model, discriminator, optimizer, source_dataloader_iter,
                                 target_dataloader_iter, val_dataloader, start_iter, last_mIoU)

    print("finished training, the best mIoU is: " + str(best_mIoU) + " in iteration " + str(best_iter))


if __name__ == '__main__':
    main()
