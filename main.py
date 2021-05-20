import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lamb import Lamb

import os
from config import load_config
from preprocess import load_data
from model import MLPMixer, ResMLP, gMLP


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def save_checkpoint(best_acc, model, optimizer, args, epoch):
    print('Best Model Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join('checkpoints2', 'checkpoint_model_best.pth'))


def _train(epoch, train_loader, model, optimizer, criterion, args):
    model.train()

    losses = 0.
    acc = 0.
    total = 0.
    for idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # batch_size, C, width, height = data.size()
        # n = (width * height) // args.patch_size ** 2
        # data = F.unfold(data, kernel_size=args.patch_size, stride=args.patch_size).view(batch_size, 7, 7, -1)

        output = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        optimizer.zero_grad()
        loss = criterion(output, target)
        losses += loss
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        if idx % args.print_intervals == 0 and idx != 0:
            print('[Epoch: {0:4d}], Loss: {1:.3f}, Acc: {2:.3f}, Correct {3} / Total {4}'.format(epoch,
                                                                                                 losses / (idx + 1),
                                                                                                 acc / total * 100.,
                                                                                                 acc, total))


def _eval(epoch, test_loader, model, args):
    model.eval()

    acc = 0.
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # batch_size, C, width, height = data.size()
            # n = (width * height) // args.patch_size ** 2
            # data = F.unfold(data, kernel_size=args.patch_size, stride=args.patch_size).view(batch_size, 7, 7, -1)
            output = model(data)
            _, pred = F.softmax(output, dim=-1).max(1)

            acc += pred.eq(target).sum().item()
        print('[Epoch: {0:4d}], Acc: {1:.3f}'.format(epoch, acc / len(test_loader.dataset) * 100.))

    return acc / len(test_loader.dataset) * 100.


def main(args):
    train_loader, test_loader = load_data(args)
    n = (args.size * args.size) // args.patch_size ** 2
    # model = MLPMixer(n, 3, args.dims, N=args.N)
    # model = ResMLP(args.dims, N=args.N)
    model = gMLP(49, args.dims, args.dims * 4, N=8)
    print('model parameters: {}'.format(get_n_params(model)))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .999))

    if not os.path.isdir('checkpoints2'):
        os.mkdir('checkpoints2')

    if args.checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', args.checkpoints))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['global_epoch']
        print("Best acc: {}".format(checkpoints['best_acc']))
    else:
        start_epoch = 1

    if args.cuda:
        model = model.cuda()

    if not args.evaluation:
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)

        global_acc = 0.
        for epoch in range(start_epoch, args.epochs + 1):
            _train(epoch, train_loader, model, optimizer, criterion, args)
            best_acc = _eval(epoch, test_loader, model, args)
            if global_acc < best_acc:
                global_acc = best_acc
                save_checkpoint(best_acc, model, optimizer, args, epoch)

            # lr_scheduler.step()
            adjust_learning_rate(optimizer, epoch, args)
            # print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
    else:
        _eval(start_epoch, test_loader, model, args)


if __name__ == '__main__':
    args = load_config()
    main(args)
