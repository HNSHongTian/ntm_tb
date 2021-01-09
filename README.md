# ntm_tb
肺结核（tb）和非结核分枝杆菌（ntm）两种病灶的二分类问题，初始数据集为tb、ntm各40个的3d nii.gz扫描图，考虑3d-resnet、2d-resnet、2d-inception-v3进行实验对比，考量分类结果
项目由三个子功能文件夹构成，3d_lung_class用来将3d的CT扫描进行病理分类；dataTransform用来将3d转为2d；2d_lung_class用于2d的CT扫描切片病理分类。
## 数据集以及模型下载

 - 下载链接：
 - 将相应数据文件夹（数据集以及模型）放到与项目相同的文件目录下

## 使用3D-ResNet进行分类
### 运行

```code
cd lung_3d_classify
python train2.py 
python train2.py --resume
```
## 3D的CT图像转为2D切片png图像

```code
cd datatransform
python getData.py
```
## 2D的CT切片分类

```code
cd tb_ntm
python train2.py
```
### 生成grad_cam

```code
cd tb_ntm
python val.py
python val_incept.py
```

