import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision
import resnet




os.environ["KMP_DUPLICATE_LIB_OK"]="True"
def draw_cam(img_path, save_path, transform=None, visheadmap=False):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0)
    model = resnet.resnet50()
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('./checkpoint/resnet.net')

    # checkpoint = torch.load('./checkpoint/resnet.net')
    model.load_state_dict(checkpoint['net'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = img.to(device)
    # model.eval()
    x = model.module.conv1(x)
    x = model.module.bn1(x)
    x = model.module.relu(x)
    x = model.module.maxpool(x)
    x = model.module.layer1(x)
    x = model.module.layer2(x)
    x = model.module.layer3(x)
    x = model.module.layer4(x)
    features = x                #1x2048x7x7
    print(features.shape)
    output = model.module.avgpool(x)   #1x2048x1x1
    print(output.shape)
    output = output.view(output.size(0), -1)
    print(output.shape)         #1x2048
    output = model.module.fc(output)   #1x1000
    print(output.shape)
    def extract(g):
        global feature_grad
        feature_grad = g
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    features.register_hook(extract)
    pred_class.backward()
    greds = feature_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(2048):
        features[i, ...] *= pooled_grads[i, ...]
    headmap = features.cpu().detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)

    if visheadmap:
        plt.matshow(headmap)
        # plt.savefig(headmap, './headmap.png')
        plt.show()

    img = cv2.imread(img_path)
    headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
    headmap = np.uint8(255*headmap)
    headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
    superimposed_img = headmap*0.4 + img
    cv2.imwrite(save_path, superimposed_img)

if __name__ == '__main__':
    # net = torchvision.models.resnet50()
    # net = net.cuda()
    # net = torch.nn.DataParallel(net)
    # checkpoint = torch.load('./checkpoint/resnet_2d.net')
    #
    # # checkpoint = torch.load('./checkpoint/resnet.net')
    # net.load_state_dict(checkpoint['net'])

    # img_path = "data/NTM_test/person17_1.png"
    # img = Image.open(img_path)
    # D = np.array(img)
    # print(type(D))
    # print(D.shape)
    transform = transforms.Compose([
        transforms.Resize(size=299),
        transforms.ToTensor(),
        transforms.Normalize((0.4069, 0.5532, 0.5352), (0.1396, 0.2267, 0.2410)),  # for jujube dataset
    ])

    draw_cam('data/NTM_test/person56_1.png', 'data/NTM_test/cam0001.png', transform=transform, visheadmap=True)
