from __future__ import print_function
import os
import sys
import importlib
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
import numpy as np
import random
from torch.utils.data import DataLoader
from utils import cal_loss, IOStream
import sklearn.metrics as metrics

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.model):
        os.makedirs('checkpoints/'+args.model)
    if not os.path.exists('checkpoints/'+args.model+'/'+'models'):
        os.makedirs('checkpoints/'+args.model+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+ args.model +'/'+'main.py.backup')
    os.system(f'cp models/{args.model}.py checkpoints' + '/' + args.model + '/' + f'model.py.backup')
    os.system('cp utils.py checkpoints' + '/' + args.model + '/' + 'utils.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.model + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=4,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=4,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model().to(device)
    print(f'#parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model = nn.DataParallel(model) 
    print(f'Experiment Model: {args.model}')
    print(str(model))
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss
    best_test_acc = 0

    for epoch in range(args.epochs):
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        for data, label in (train_loader):
            data, label = data.to(device), label.to(device).squeeze() 

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            

        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                train_loss*1.0/count,
                                                                                metrics.accuracy_score(
                                                                                train_true, train_pred),
                                                                                metrics.balanced_accuracy_score(
                                                                                train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = []
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for test_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            if epoch == 0 and test_idx > 10 and len(data) == args.test_batch_size:
                # Time the forward pass.
                starter.record()
                logits = model(data)
                ender.record()
                torch.cuda.synchronize()
                total_time.append(starter.elapsed_time(ender))
            else:
                logits = model(data)

            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        if epoch == 0:
            print(f'test total time batch statistics: {total_time}.')
            print(f'test total time is mean/std/min/max: {np.mean(total_time)}/{np.std(total_time)}/{np.min(total_time)}/{np.max(total_time)}.')

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.model)

    print(f"Best test acc: {best_test_acc}")


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model(args).to(device)
    model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1] 
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--model', type=str, required=True,
                        help='model to use, [ease_PCT, PCT, ease_pointnet2, pointnet2]')
    args = parser.parse_args()

    _init_()
    seed_everything(args.seed)
    sys.path.append('./models')

    io = IOStream('checkpoints/' + args.model + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)