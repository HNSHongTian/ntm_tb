# ntm_tb
肺结核（tb）和非结核分枝杆菌（ntm）两种病灶的二分类问题，初始数据集为tb、ntm各40个的3d nii.gz扫描图，考虑3d-resnet、2d-resnet、2d-inception-v3进行实验对比，考量分类结果
项目由三个子功能文件夹构成，3d_lung_class用来将3d的CT扫描进行病理分类；dataTransform用来将3d转为2d；2d_class_lung用于2d的CT扫描切片病理分类。
## 数据集以及模型下载

 - 下载链接： https://pan.baidu.com/s/1JilB_684Z5qnLkN_ppMpiw 提取码: ytaf  
 - 将相应数据文件夹（数据集以及模型）放到与项目相同的文件目录下
 - 删除所有的cpp文件

## 使用3D-ResNet进行分类
### 运行

```code
cd 3d_lung_class
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
cd 2d_class_lung
python train2.py
```
### 生成grad_cam

```code
cd 2d_class_lung
python val.py
python val_incept.py
```

