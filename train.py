import torch
from torch.autograd import Variable
from torch import optim, nn
from shuffle_net.shuffle_net_alt import ShuffleNet
import argparse
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from mobile_net.mobile_net import MobileNet


def train(model, dataloader, max_epoch, base_lr,
          lr_decay, weight_decay, use_gpu, print_every):
    optimizer = optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    for ep in range(max_epoch):
        for idx, (data, label) in enumerate(dataloader):
            data, label = Variable(data).type(FloatTensor), Variable(label).type(LongTensor)
            y_pred = model(data)
            loss = criterion(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % print_every == 0:
                print('{} Epoch: loss is {}'.format(ep, loss.data[0]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay
        torch.save(model.state_dict(), './checkpoint/ckpt.pth')


def validate(model, dataloader, use_gpu):
    criterion = nn.CrossEntropyLoss()
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training light model')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.999, help='decay learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--device', type=int, default=0, help='gpu device')
    parser.add_argument('--max_epoch', type=int, default=1000, help='max epochs')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', type=bool, default=False)

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(args.device)

    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch

    print('==> Preparing data..')
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_root = '/home/liuhanqing/dataset/train/'
    val_root = '/home/liuhanqing/dataset/val/'
    train_dataset = datasets.ImageFolder(
        train_root,
        train_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_dataset = datasets.ImageFolder(
        val_root,
        val_transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = MobileNet().cuda() if use_gpu else MobileNet()
    train(model, )

