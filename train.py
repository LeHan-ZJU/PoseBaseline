"""
# --------------------------------------------------------
# @Project: MyPoseNet
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2021-06-17
# --------------------------------------------------------
"""

import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from Models.eval_pose import eval_net
from Models.RatNet import Net_Resnet
from utils.dataset_csv import DatasetPoseCSV
from torch.utils.data import DataLoader, random_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dir_img = ''  # data file
dir_label = ''   # label file(csv)
dir_checkpoint = './TrainedModel/'
num_points = 6

resize_w = 320
resize_h = 256
extract_list = ["layer4"]

isExists = os.path.exists(dir_checkpoint)
if not isExists:  # 判断结果
    os.makedirs(dir_checkpoint)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target heatmaps',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=3,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str,  default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=4,
                        help='the ratio between img and GT')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-p', '--path_backbone', dest='path_backbone', type=str,  # default=None,
                        default='./Resnet/resnet50-19c8e357.pth',
                        help='the path of backbone')
    return parser.parse_args()

def train_net(net,
              device,
              epochs=30,
              batch_size=4,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1
              ):   # scale是输入与输出的边长比

    dataset = DatasetPoseCSV(resize_w, resize_h, dir_img, dir_label, img_scale, num_points)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # if net.n_classes > 1 else 'max', patience=2)

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_heatmaps = batch['heatmap']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                heatmap_type = torch.float32  # if net.n_classes == 1 else torch.long
                true_heatmaps = true_heatmaps.to(device=device, dtype=heatmap_type)

                heatmaps_pred = net(imgs)

                loss_mse = criterion(heatmaps_pred, true_heatmaps)
                loss = loss_mse

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:

                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))

        print('epoch:', epoch + 1, ' loss:', loss.item())

        if (epoch-9) % 10 == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    return loss_all


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    print(args.path_backbone)
    print('input_size:', resize_w, resize_h, ';  Augment:', args.augment)
    print('lr:', args.lr, ';  batch_size:', args.batchsize, ';  the weight of similar loss:', args.weight)
    print('trainset:', dir_label)

    # 构建网络
    net = Net_Resnet(args.path_backbone, extract_list, device, train=True, n_channels=3, nof_joints=num_points)
    # print(net)

    if args.load:
        print(args.load)
        net.load_state_dict(
            torch.load(args.load, map_location=device), strict=False   #strict，该参数默认是True，表示预训练模型的层和自己定义的网络结构层严格对应相等（比如层名和维度）
        )
        logging.info(f'Model loaded from {args.load}')
        print('Pretrained weights have been loaded!')
    else:
        print('No pretrained models have been loaded except the backbone!')

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        loss_all = train_net(net=net,
                             device=device,
                             epochs=args.epochs,
                             batch_size=args.batchsize,
                             lr=args.lr,
                             val_percent=args.val / 100,
                             img_scale=args.scale
                             )


    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
