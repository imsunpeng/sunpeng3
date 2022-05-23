import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join


from torch import Tensor
class LITS_preprocess:
    def __init__(self, raw_dataset_path,fixed_dataset_path, ):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes =3
        self.upper = 200
        self.lower = -200
        self.expand_slice = 1  # 轴向外侧扩张的slice数量
        self.xy_down_scale = 0.5
        self.slice_down_scale = 1

        self.valid_rate = 0.1

    def fix_data(self):
        if not os.path.exists(self.fixed_path):    # 创建保存目录
            os.makedirs(join(self.fixed_path))
            #os.makedirs(join(self.fixed_path))
        file_list = os.listdir(join(self.raw_root_path))
        Numbers = len(file_list)
        print('Total numbers of samples is :',Numbers)
        for ct_file,i in zip(file_list,range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i+1,Numbers))

            ct_path = os.path.join(self.raw_root_path,  ct_file)
            seg_path = os.path.join(self.raw_root_path,  ct_file.replace('imaging', 'segmentation'))
            new_ct, new_seg = self.process(ct_path, seg_path, classes = self.classes)
            if new_ct != None and new_seg != None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path,  ct_file))
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path,  ct_file.replace('imaging', 'segmentation')))

    def process(self, ct_path, seg_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:",ct_array.shape, seg_array.shape)


        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > self.upper] = self.upper
        ct_array[ct_array < self.lower] = self.lower

        # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
        ct_array = ndimage.zoom(ct_array,
                                (self.xy_down_scale, self.xy_down_scale, ct.GetSpacing()[-1] / self.slice_down_scale),
                                order=3)
        seg_array = ndimage.zoom(seg_array,
                                 (self.xy_down_scale, self.xy_down_scale, ct.GetSpacing()[-1] / self.slice_down_scale),
                                 order=0)

        # 找到肝脏区域开始和结束的slice，并各向外扩张
        z = np.any(seg_array > 0, axis=(0, 1))
        #print(z)
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        #print (start_slice, end_slice )



        # 两个方向上各扩张个slice
        if start_slice - self.expand_slice < 0:
            start_slice = 0
        else:
            start_slice -= self.expand_slice

        if end_slice + self.expand_slice >= seg_array.shape[2]:
            end_slice = seg_array.shape[2] - 1
        else:
            end_slice += self.expand_slice


        print("Cut out range:",str(start_slice) + '--' + str(end_slice))
        # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少

        # 截取保留区域
        ct_array = ct_array[ :, :,start_slice:end_slice + 1]
        seg_array = seg_array[ :, :,start_slice:end_slice + 1]

       # ct_array = Tensor(ct_array)
        #seg_array = Tensor(seg_array)
        # z.shape is (3, 2, 5, 6)


        #ct_array =ct_array.transpose(0, 2)
        #seg_array = seg_array.transpose(0, 2)


        print("Preprocessed shape:",ct_array.shape,seg_array.shape)
        # 保存为对应的格式
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))

        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        return new_ct, new_seg

    def write_train_val_name_list(self):
        data_name_list = os.listdir(join(self.fixed_path))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num*(1-self.valid_rate))]
        val_name_list = data_name_list[int(data_num*(1-self.valid_rate)):int(data_num*((1-self.valid_rate) + self.valid_rate))]

        #self.write_name_list(train_name_list, "train_path_list.txt")
        #self.write_name_list(val_name_list, "val_path_list.txt")


    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, name)
            seg_path = os.path.join(self.fixed_path,  name.replace('imaging', 'segmentation'))
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()

if __name__ == '__main__':
    for i in range(300):


        if i<10:
            raw_dataset_path = './3d_data/data/'+'case_0000'+str(i)
            print(raw_dataset_path)
            fixed_dataset_path = './kits21/data/'+'case_0000'+str(i)

            tool = LITS_preprocess(raw_dataset_path,fixed_dataset_path)
            tool.fix_data()                            # 对原始图像进行修剪并保存
            tool.write_train_val_name_list()      # 创建索引txt文件
        elif i < 100:
            raw_dataset_path = './3d_data/data/' + 'case_000'+str(i)
            print(raw_dataset_path)
            fixed_dataset_path = './kits21/data/' + 'case_000' + str(i)

            tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path)
            tool.fix_data()  # 对原始图像进行修剪并保存
            tool.write_train_val_name_list()  # 创建索引txt文件
        else:
            raw_dataset_path = './3d_data/data/' + 'case_00' + str(i)
            print(raw_dataset_path)
            fixed_dataset_path = './kits21/data/' + 'case_00'+str(i)

            tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path)
            tool.fix_data()  # 对原始图像进行修剪并保存
            tool.write_train_val_name_list()  # 创建索引txt文件
