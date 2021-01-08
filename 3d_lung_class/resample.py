import os
import numpy as np
import SimpleITK as sitk
import torch

def load_data(image_path):
    return sitk.ReadImage(image_path)


def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    # print("----------------------------")
    # print('original_spacing is: {}'.format(original_spacing))
    # print('original_size is: {}'.format(original_size))
    # print('original image is:')
    # print(sitk.GetArrayFromImage(itk_image))
    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    # print('out_size is: {}'.format(out_size))

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    output = resample.Execute(itk_image)
    # print('after resample output: type:{}'.format( type(output)))
    # print('output image is: ')
    # print(sitk.GetArrayFromImage(output))
    # y = torch.Tensor(output)
    # print("----------------------------")
    return output


def check_artifact(image_agency, label_agency, name):

    label = sitk.GetArrayFromImage(label_agency)

    # check first several slices
    if label[0, 0, 0] != 0 and np.sum(label[0] - label[0, 0, 0]) == 0:
        print("detected artifact in the first slice", name)

        count = 0
        while label[count, 0, 0] != 0 and np.sum(label[count] - label[count, 0, 0]) == 0:
            count += 1

        label = label[count:, :, :]
        label_agency = sitk.GetImageFromArray(label)

        image = sitk.GetArrayFromImage(image_agency)
        image = image[count:, :, :]
        assert image.shape == label.shape
        image_agency = sitk.GetImageFromArray(image)

    z_max = label.shape[0] - 1

    # check last several slices
    if label[z_max, 0, 0] != 0 and np.sum(label[z_max] - label[z_max, 0, 0]) == 0:
        print("detected artifact in the last slice", name)

        count = z_max
        while label[count, 0, 0] != 0 and np.sum(label[count] - label[count, 0, 0]) == 0 and z_max > 0:
            count -= 1

        label = label[:count, :, :]
        label_agency = sitk.GetImageFromArray(label)

        image = sitk.GetArrayFromImage(image_agency)
        image = image[:count, :, :]
        assert image.shape == label.shape
        image_agency = sitk.GetImageFromArray(image)

    return image_agency, label_agency


def square_pad(image_agency):
    size = image_agency.GetSize()

    if size[0] == size[1]:
        return image_agency
    else:
        print('applying square padding')

        image = sitk.GetArrayFromImage(image_agency)  # shape: (channel, x, y)

        if image.shape[1] > image.shape[2]:
            length = (image.shape[1] - image.shape[2]) / 2
            image = np.pad(image, pad_width=((0, 0), (0, 0), (int(np.ceil(length)), int(np.floor(length)))),
                           mode='constant', constant_values=(np.min(image),))
        else:
            length = (image.shape[2] - image.shape[1]) / 2
            image = np.pad(image, pad_width=((0, 0), (int(np.ceil(length)), int(np.floor(length))), (0, 0)),
                           mode='constant', constant_values=(np.min(image),))

        assert image.shape[1] == image.shape[2], 'pad failed'
        image_agency = sitk.GetImageFromArray(image)
        return image_agency


if __name__ == '__main__':

    dataset_root = 'data_test/NTM_test'
    # target_directory = '/mnt/data/TB_resampled'
    imgs = []
    labels = []

    for patient in os.listdir(dataset_root):
        if patient.startswith('.'): continue

        # if not os.path.exists(os.path.join(target_directory, patient)):
        #     os.makedirs(os.path.join(target_directory, patient))

        for case in os.listdir(os.path.join(dataset_root, patient)):
            if case.startswith('.'): continue

            data = os.listdir(os.path.join(dataset_root, patient, case))
            assert len(data) == 2

            # if not os.path.exists(os.path.join(target_directory, patient, case)):
            #     os.makedirs(os.path.join(target_directory, patient, case))

            label_path, image_path = None, None

            for dicom_series in data:
                if 'label' in dicom_series:
                    label_path = os.path.join(dataset_root, patient, case, dicom_series)
                else:
                    image_path = os.path.join(dataset_root, patient, case, dicom_series)
            assert label_path is not None and image_path is not None

            image = load_data(image_path)
            image = square_pad(image)
            # print("this img out")
            # print(image)
            image = resample_image(image, out_spacing=(1.0, 1.0, 1.0))
            print("this img output type----------")
            # print(image)
            img = sitk.GetArrayFromImage(image)
            print(type(img))
            o = torch.Tensor(img)
            print("this img output type----------")
            print(type(o))
            print(o.shape)
            print(o)
            label = load_data(label_path)
            label = square_pad(label)
            label = resample_image(label, out_spacing=(1.0, 1.0, 1.0), is_label=True)

            image, label = check_artifact(image, label, patient + '_' + case)
            print(image)
            print(label)
            imgs.append(image)
            labels.append(labels)
    print(len(imgs))
    i= 0
    for img in imgs:
        print("------")
        # img = sitk.GetArrayFromImage(img)
        # out = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, 'dataTest/NTM/save_%d.nii.gz'%i)
        i = i + 1
        # img = torch.Tensor(img)
        # print(img.size())
        print("------")


            # sitk.WriteImage(image, os.path.join(target_directory, patient, case, image_path.split('/')[-1]))
            # sitk.WriteImage(label, os.path.join(target_directory, patient, case, label_path.split('/')[-1]))
