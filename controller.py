import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

import numpy as np
import time
import os

from models import resnet, alexnet
import util
from logger import Logger
import matplotlib.pyplot as plt

print('Loading data')

transform_train = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.Resize(224),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_modelnet = False

resume = False
only_test = False

if use_modelnet:
    from custom_dataset2 import MultiViewDataSet
    dset_train = MultiViewDataSet('../MVCNN-PyTorch/classes', 'train', transform=transform_train)
    train_loader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=2)

    dset_val = MultiViewDataSet('../MVCNN-PyTorch/classes', 'val', transform=transform_test)
    val_loader = DataLoader(dset_val, batch_size=4, shuffle=True, num_workers=2)

    dset_test = MultiViewDataSet('../MVCNN-PyTorch/classes', 'test', transform=transform_test)
    test_loader = DataLoader(dset_test, batch_size=4, shuffle=True, num_workers=2)


else:
    from custom_dataset import MultiViewDataSet
    dset_train = MultiViewDataSet('../datasets/all/train', transform=transform_train)
    train_loader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=2)

    dset_val = MultiViewDataSet('../datasets/all/val', transform=transform_test)
    val_loader = DataLoader(dset_val, batch_size=4, shuffle=False, num_workers=2)

    dset_test = MultiViewDataSet('../datasets/all/test', transform=transform_test)
    test_loader = DataLoader(dset_test, batch_size=4, shuffle=False, num_workers=2)


classes = dset_train.classes
print(len(dset_train.x), len(dset_val.x), len(dset_test.x))
print(len(classes), classes)

train_loss = []
train_acc = []
val_loss = []
val_acc = []

resnet = resnet.resnet18(num_classes=len(classes))
resnet.to(device)

#resnet = alexnet.mvcnn(pretrained=True)
#resnet.to(device)

cudnn.benchmark = True

print(device)

logger = Logger('logs')

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.0001
# optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
optimizer = torch.optim.SGD(
    resnet.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4
)

n_epochs = 25

best_acc = 0.0
best_loss = 0.0
start_epoch = 0


# Helper functions
def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile('checkpoint/checkpoint.pth.tar'), 'Error: no checkpoint file found!'

    checkpoint = torch.load('checkpoint/checkpoint.pth.tar')
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    resnet.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict
    global optimizer, lr
    optimizer = torch.optim.SGD(
        resnet.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )


def train():
    print(lr)
    train_size = len(train_loader)

    total_loss = 0.0

    total = 0.0
    correct = 0.0
    n = 0

    for i, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1)

        #plt.imshow(np.transpose(inputs[0][0], (1, 2, 0)))
        #plt.show()

        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        # compute output
        outputs = resnet(inputs)
        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted.cpu() == targets.cpu()).sum()
        total_loss += loss.item()
        n += 1

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss.item()))

    avg_acc = 100 * correct / total

    train_loss.append(total_loss / len(train_loader))
    train_acc.append(avg_acc)

# Validation and Testing
def eval(data_loader, is_test=False):
    if is_test:
        load_checkpoint()

    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    start = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            outputs = resnet(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    time_taken = time.time() - start
    print('Time: ', time_taken)
    print('Avg: ', time_taken / len(data_loader)*1000, 'ms')

    avg_acc = 100 * correct / total
    avg_loss = total_loss / len(val_loader)

    val_loss.append(avg_loss)
    val_acc.append(avg_acc)

    return avg_acc, avg_loss


# Testing

if only_test:
    resnet.eval()
    avg_test_acc, avg_loss = eval(test_loader, is_test=True)

    print('\nTest:')
    print('\tTest Acc: %.2f - Loss: %.4f' % (avg_test_acc.item(), avg_loss.item()))
    print('\tModel val acc: %.2f' % best_acc)
    exit(0)

# Training / Eval loop

if resume:
    load_checkpoint()

for epoch in range(start_epoch, n_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    resnet.train()
    train()
    print('Time taken: %.2f sec.' % (time.time() - start))

    resnet.eval()
    avg_test_acc, avg_loss = eval(val_loader)

    print('\nEvaluation:')
    print('\tVal Acc: %.2f - Loss: %.4f' % (avg_test_acc.item(), avg_loss.item()))
    print('\tCurrent best val acc: %.2f' % best_acc)

    util.logEpoch(logger, resnet, epoch + 1, avg_loss, avg_test_acc)

    # Save model
    if avg_test_acc > best_acc:
        print('\tSaving checkpoint - Acc: %.2f' % avg_test_acc)
        best_acc = avg_test_acc
        best_loss = avg_loss
        util.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': resnet.state_dict(),
            'acc': avg_test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        })

    if (epoch + 1) % 20 == 0:
        print('Changed to batch size 20')
        train_loader = DataLoader(dset_train, batch_size=20, shuffle=True, num_workers=2)
        val_loader = DataLoader(dset_val, batch_size=20, shuffle=False, num_workers=2)

    # Decaying Learning Rate
    #if (epoch + 1) % 5 == 0:
    #    lr *= 0.5
    #    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    #    print('Learning rate:', lr)

fig = plt.figure(1)
ax = fig.add_subplot(111)

plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
ax.plot(train_loss,'g--^', label='Train Loss')
ax.plot(val_loss, 'r--o', label='Validation Loss')

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))

fig.savefig('mvcnn_loss', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

plt.close(fig)

fig2 = plt.figure(1)
ax2 = fig2.add_subplot(111)
plt.ylim(0, 105)
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
ax2.plot(train_acc , 'g--^', label='Train Accuracy')
ax2.plot(val_acc, 'r--o', label='Validation Accuracy')

handles, labels = ax2.get_legend_handles_labels()
lgd = ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))

fig2.savefig('mcvnn_acc', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
