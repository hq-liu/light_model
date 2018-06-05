import argparse
import os
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
from torchvision import transforms
from torch.nn import functional as F
from new_model.new_model import NewModel
from torchvision import datasets


parser = argparse.ArgumentParser()
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float)
parser.add_argument('--lr_decay', default=0.97, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--bs', '--batch_size', default=64, type=int)
parser.add_argument('--wd', '--weight_decay', default=0.004, type=float)
parser.add_argument('--e', '--epochs', default=50, type=int)
parser.add_argument('--r', '--resume', default=False, type=bool)
parser.add_argument('--cn', '--checkpoint_name', default='by_pass_net.pth', type=str)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--print-freq', '-p', default=10, type=int)


def main():
    global args
    args = parser.parse_args()
    model = NewModel()

    use_gpu = torch.cuda.is_available()
    global FloatTensor, LongTensor
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor

    if use_gpu:
        model = torch.nn.DataParallel(model).to("cuda")

    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = data.DataLoader(train_dataset, batch_size=args.bs,
                                   shuffle=True, num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.bs, shuffle=False,
        num_workers=8, pin_memory=True)

    for epoch in range(args.e):
        # train for one epoch
        train(model, train_loader, criterion, optimizer, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * args.lr_decay

        # evaluate on validation set
        validate(val_loader, model, criterion)
        #
        # # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best)


def train(model, train_loader, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (data, label_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data = data.type(FloatTensor)
        label_target = label_target.type(LongTensor)

        output = model(data)

        loss = criterion(output, label_target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, label_target, topk=(1, 5))
        bs = data.size(0)
        losses.update(loss.item(), bs)
        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        if i % (args.print_freq * 10) == 0:
            torch.save(model.state_dict(), './checkpoint/' + args.cn)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.type(FloatTensor)
        target = target.type(LongTensor)

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        bs = data.size(0)
        losses.update(loss.item(), bs)
        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

