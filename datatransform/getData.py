import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
import torchvision.utils as vutils

def getDataOFDataset(path):
    for root, ds,fs in os.walk(path):
        for f in fs:
            if f.endswith('.nii.gz') and "label" not in f:
                fullname = os.path.join(root, f)
                yield fullname



def getDataLabel(path):
    for root, ds,fs in os.walk(path):
        for f in fs:
            if f.endswith('.nii.gz') and "label" in f:
                fullname = os.path.join(root, f)
                yield fullname


def getTotalData(path):
    img = []
    for i in getDataLabel(path):
        img_label = i
        img_org = i.replace("-label", "")
        # print(img_org)
        # print(img_label)
        img.append((img_org, img_label))
    return img



def calculateSlice(path):
    images = getTotalData(path)
    j = 0
    for img, label in images:
        j += 1
        itk_img = sitk.ReadImage(label)
        label_img = sitk.GetArrayFromImage(itk_img)
        itk_img2 = sitk.ReadImage(img)
        org_img = sitk.GetArrayFromImage(itk_img2)
        org_img[org_img < -1200] = -1200
        org_img[org_img > 1200] = 1200
        (z, x, y) = label_img.shape
        num_slice = 0
        for i in range(z):
            if np.sum(label_img[i]) != 0:
                num_slice += 1
                # img_slice = (org_img[i] - np.min(org_img[i])) / (np.max(org_img[i]) - np.min(org_img[i])) * 255
                img_label_slice = (label_img[i] - np.min(label_img[i])) / (np.max(label_img[i]) - np.min(label_img[i])) * 255
                save_img = Image.fromarray(np.uint8(img_label_slice))
                save_img.save(os.path.join("./data2d", path, "person" + str(j) + "_"+str(num_slice) + '_label.png'), "PNG")





calculateSlice("Tuberculosis")