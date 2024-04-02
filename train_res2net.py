import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from net.birnet_res2net import Net
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np

file = open("log/BIRNet.txt", "a")
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
torch.backends.cudnn.benchmark = True


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    predict = torch.sigmoid(predict)
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch):
    model.train()

    loss_record4, loss_record3, loss_record2, loss_record1, loss_recorde4, loss_recorde3, loss_recorde2, loss_recorde1 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()
        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, edge_map_4, edge_map_3, edge_map_2, edge_map_1 = model(
            images)
        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss1 = structure_loss(lateral_map_1, gts)
        losse4 = dice_loss(edge_map_4, edges)
        losse3 = dice_loss(edge_map_3, edges)
        losse2 = dice_loss(edge_map_2, edges)
        losse1 = dice_loss(edge_map_1, edges)
        loss = 0.25*loss4 + 0.5*loss3 + 0.75*loss2 + loss1 + 0.25*losse4 + 0.5*losse3 + 0.75*losse2 + losse1
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        loss_record4.update(loss4.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        loss_recorde4.update(losse4.data, opt.batchsize)
        loss_recorde3.update(losse3.data, opt.batchsize)
        loss_recorde2.update(losse2.data, opt.batchsize)
        loss_recorde1.update(losse1.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-4: {:.4f}], [lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge-4: {:,.4f}], [edge-3: {:,.4f}], [edge-2: {:,.4f}], [edge-1: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record4.avg, loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde4.avg,
                         loss_recorde3.avg, loss_recorde2.avg, loss_recorde1.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-4: {:.4f}], [lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge-4: {:,.4f}], [edge-3: {:,.4f}], [edge-2: {:,.4f}], [edge-1: {:,.4f}]'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                              loss_record4.avg, loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde4.avg,
                              loss_recorde3.avg, loss_recorde2.avg, loss_recorde1.avg))

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'BIRNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'BIRNet-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'BIRNet-%d.pth' % epoch + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=20, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=448, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='BIR_res2net')
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = './DATA/TrainDataset/Imgs/'.format(opt.train_path)
    gt_root = './DATA/TrainDataset/GT/'.format(opt.train_path)
    edge_root = './DATA/TrainDataset/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)

    file.close()
