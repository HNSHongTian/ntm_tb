from __future__ import print_function
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import SimpleITK as sitk
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import resample
import os
import argparse
import resnet3d
import torch.backends.cudnn as cudnn



def getData(dataset_root):


    imgs = []
    labels = []
    i = 0
    for patient in os.listdir(dataset_root):
        if patient.startswith('.'): continue



        for case in os.listdir(os.path.join(dataset_root, patient)):
            if case.startswith('.'): continue

            data = os.listdir(os.path.join(dataset_root, patient, case))
            assert len(data) == 2



            label_path, image_path = None, None

            for dicom_series in data:
                if 'label' in dicom_series:
                    label_path = os.path.join(dataset_root, patient, case, dicom_series)
                else:
                    image_path = os.path.join(dataset_root, patient, case, dicom_series)
            assert label_path is not None and image_path is not None

            image = resample.load_data(image_path)
            image = resample.square_pad(image)
            image = resample.resample_image(image, out_spacing=(1.0, 1.0, 1.0))
            img = sitk.GetArrayFromImage(image)
            o = torch.Tensor(img)
            label = resample.load_data(label_path)
            label = resample.square_pad(label)
            label = resample.resample_image(label, out_spacing=(1.0, 1.0, 1.0), is_label=True)

            image, label = resample.check_artifact(image, label, patient + '_' + case)
            imgs.append(image)
            labels.append(labels)

    return imgs





def getData_img(path1, path2):
    imgs = []
    paths = os.listdir(path1)
    for p in paths:
        ip = os.path.join(path1, p)
        img = sitk.GetArrayFromImage(sitk.ReadImage(ip))
        img[img < -1200] = -1200
        img[img > 1200] = 1200
        imgs.append((img, 1))
    paths2 = os.listdir(path2)
    for p in paths2:
        ip = os.path.join(path2, p)
        img = sitk.GetArrayFromImage(sitk.ReadImage(ip))
        img[img < -1200] = -1200
        img[img > 1200] = 1200
        imgs.append((img, 0))
    return imgs


transforms = transforms.Compose([
    # transforms.Resize(256),
    # transforms.ToTensor,
    transforms.Normalize(0.5, 0.5)
])


class Train_Dataset(Dataset):
    def __init__(self, transform=None):
        self.imgs = getData_img("data/NTM", "data/TB")
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_np, label = self.imgs[idx]
        img_np = torch.Tensor(img_np)
        img_np = img_np.unsqueeze(0)
        img_np = img_np.unsqueeze(0)
        img_np = F.interpolate(img_np, size=(256, 256, 256), mode='trilinear')
        img_np = img_np.squeeze(0)
        print(img_np.size())
        return img_np, label


class Test_Dataset(Dataset):
    def __init__(self, transform=None):
        self.imgs = getData_img("dataTest/NTM", "dataTest/TB")
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_np, label = self.imgs[idx]
        img_np = torch.Tensor(img_np)
        img_np = img_np.unsqueeze(0)
        img_np = img_np.unsqueeze(0)
        img_np = F.interpolate(img_np, size=(256, 256, 256), mode='trilinear')
        img_np = img_np.squeeze(0)
        print(img_np.size())
        return img_np, label


parser = argparse.ArgumentParser(description='PyTorch jujube Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(size=299, scale=(0.7, 1.0), ratio=(3./4., 4./3.)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4069, 0.5532, 0.5352), (0.1396, 0.2267, 0.2410)), # for jujube dataset
# ])
#
# transform_test = transforms.Compose([
#     transforms.Resize(size=299),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4069, 0.5532, 0.5352), (0.1396, 0.2267, 0.2410)), # for jujube dataset
# ])
#
# # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainset = torchvision.datasets.ImageFolder(root='data', transform=transform_train)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
#
# # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testset = torchvision.datasets.ImageFolder(root='data2d', transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
#
# classes = ('ntm','tb')
liver_dataset = Train_Dataset()
dataloaders = DataLoader(liver_dataset, batch_size=1, shuffle=True, num_workers=4)
liver_dataset2 = Test_Dataset()
dataloaders2 = DataLoader(liver_dataset2, batch_size=1, shuffle=True, num_workers=4)
print(liver_dataset.__len__())
print(liver_dataset2.__len__())
# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = resnet.resnet50()
# net = torchvision.models.resnet50()
# net = inceptionNet.InceptionV3()
# net = torchvision.models.inception_v3()
# net.aux_logits = False
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = inceptionV4.Googlenetv4()
net = resnet3d.resnet34()
# net = net.to(device)

if device == 'cuda':
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/resnet3d4.net')

    # checkpoint = torch.load('./checkpoint/resnet.net')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.1 ** (epoch // 400))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # adjust_learning_rate(optimizer, epoch, 0.1)
    # lr = optimizer.param_groups[0]['lr']
    # print(epoch, lr)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = args.lr * (0.1 ** (epoch // 100))
    #     print("changing the lr")

    # optimizer.param_groups['lr'] = args.lr * (0.1 ** (epoch // 30))
    for batch_idx, (inputs, targets) in enumerate(dataloaders):
        # inputs, targets = inputs.to(device), targets.to(device) # pytorch 0.3.0 not supported
        if device == 'cuda':
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
           % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        file = open('loss_train_inception', 'a')
        file.write('Loss: %.3f  Acc: %.3f%% (%d/%d)'
           % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloaders2):
            if device == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # print(loss)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            file = open('loss_test_inception', 'a')
            file.write('Loss: %.3f  Acc: %.3f%% (%d/%d)'
                       % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(best_acc)
    if acc > best_acc and epoch > 400:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/shufflenet.net')
        torch.save(state, './checkpoint/resnet3d4.net')
        best_acc = acc
        print(best_acc)


for epoch in range(start_epoch, start_epoch+800):
    train(epoch)
    test(epoch)

