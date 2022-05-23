
import nibabel as nib
import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib2 import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import KiTS21
from dataset.transform import MedicalTransform
from loss import GeneralizedDiceLoss
from loss.util import class2one_hot
from network import AGDenseUNet
from utils.metrics import Evaluator
from utils.vis import imshow

@click.command()
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=1, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=1, show_default=True)
@click.option('-s', '--size', 'img_size', help='Output image size', type=(int, int),
              default=(256, 256), show_default=True)
@click.option('-d', '--data', 'data_path', help='Path of kits19 data after conversion',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('-r', '--resume', help='Resume model',
              type=click.Path(exists=True, file_okay=True, resolve_path=True),
              default="runs/AGDenseUNet/checkpoint/best.pth")
@click.option('-o', '--output', 'output_path', help='output image path',
              type=click.Path(dir_okay=True, resolve_path=True), default='out', show_default=True)
@click.option('--vis_intvl', help='Number of iteration interval of display visualize image. '
                                  'No display when set to 0',
              type=int, default=20, show_default=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=0, show_default=True)
def main(batch_size, num_gpu, img_size, data_path, resume, output_path, vis_intvl, num_workers):
    data_path = Path(data_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    roi_error_range = 0
    transform = MedicalTransform(output_size=img_size, roi_error_range=0, use_roi=False)

    dataset = KiTS21(data_path, stack_num=5, spec_classes=[0, 1, 2, 3], img_size=img_size,
                     use_roi=False, roi_file='roi.json', roi_error_range=0,
                     train_transform=transform, valid_transform=transform)

    net =  AGDenseUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)

    if resume:
        data = {'net': net}
        cp_file = Path(resume)
        cp.load_params(data, cp_file, device='cpu')

    criterion = GeneralizedDiceLoss(idc=[0, 1, 2, 3])

    gpu_ids = [i for i in range(num_gpu)]

    print(f'{" Start evaluation ":-^40s}\n')
    msg = f'Net: {net.__class__.__name__}\n' + \
          f'Dataset: {dataset.__class__.__name__}\n' + \
          f'Batch size: {batch_size}\n' + \
          f'Device: cuda{str(gpu_ids)}\n'
    print(msg)

    torch.cuda.empty_cache()

    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()

    net.eval()
    torch.set_grad_enabled(False)
    transform.eval()

    valid_score = evaluation(net, dataset, batch_size, num_workers, vis_intvl, type='valid')
    print(f'Valid data score: {valid_score:.5f}')

    subset = dataset.test_dataset
    case_slice_indices = dataset.test_case_slice_indices

    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)

    case = 0
    vol_output = []

    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/test', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, idx = data['image'].cuda(), data['index']

            outputs = net(imgs)
            #predicts = outputs['output']
            predicts = outputs.argmax(dim=1)

            predicts = predicts.cpu().detach().numpy()
            idx = idx.numpy()

            vol_output.append(predicts)

            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]



                #roi = dataset.get_roi(case, type='test')
                roi = dataset.case_idx_to_case_id(case, 'test')
                vol_ = vol_output[:vol_num_slice]
                #vol_ = reverse_transform(vol, roi, dataset, transform)
                vol_ = vol_.astype(np.uint8)

                case_id = dataset.case_idx_to_case_id(case, type='test')
                affine = np.load(data_path / f'case_{case_id:05d}' / 'affine.npy')
                vol_nii = nib.Nifti1Image(vol_, affine)
                vol_nii_filename = output_path / f'prediction_{case_id:05d}.nii.gz'
                vol_nii.to_filename(str(vol_nii_filename))

                vol_output = [vol_output[vol_num_slice:]]
                case += 1
                pbar.update(1)

            if vis_intvl > 0 and batch_idx % vis_intvl == 0:
                data['predict'] = predicts
                data = dataset.vis_transform(data)
                imgs, predicts = data['image'], data['predict']
                imshow(title=f'eval/test', imgs=(imgs[0, 1], predicts[0]), shape=(1, 2),
                       subtitle=('image', 'predict'))




def evaluation(net, dataset, batch_size, num_workers,vis_intvl, type):
    type = type.lower()
    if type == 'train':
        subset = dataset.train_dataset
        case_slice_indices = dataset.train_case_slice_indices
    elif type == 'valid':
        subset = dataset.valid_dataset
        case_slice_indices = dataset.valid_case_slice_indices

    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)
    evaluator = Evaluator(dataset.num_classes)

    case = 0
    vol_label = []
    vol_output = []

    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, labels, idx = data['image'].cuda(), data['label'], data['index']

            outputs = net(imgs)
            #predicts = outputs['output']
            predicts = outputs.argmax(dim=1)

            labels = labels.cpu().detach().numpy()
            predicts = predicts.cpu().detach().numpy()
            idx = idx.numpy()

            vol_label.append(labels)
            vol_output.append(predicts)

            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_label = np.concatenate(vol_label, axis=0)

                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]
                evaluator.add(vol_output[:vol_num_slice], vol_label[:vol_num_slice])

                vol_output = [vol_output[vol_num_slice:]]
                vol_label = [vol_label[vol_num_slice:]]
                case += 1
                pbar.update(1)



            if vis_intvl > 0 and batch_idx % vis_intvl == 0:
                data['predict'] = predicts
                data = dataset.vis_transform(data)
                imgs, labels, predicts = data['image'], data['label'], data['predict']
                imshow(title=f'eval/{type:5}', imgs=(imgs[0, dataset.img_channels // 2], labels[0], predicts[0]),
                       shape=(1, 3), subtitle=('image', 'label', 'predict'))


    acc = evaluator.eval()

    for k in sorted(list(acc.keys())):
        if k == 'dc_each_case': continue
        print(f'{type}/{k}: {acc[k]:.5f}')
        #logger.add_scalar(f'{type}_acc_total/{k}', acc[k], epoch)

    for case_idx in range(len(acc['dc_each_case'])):
        case_id = dataset.case_idx_to_case_id(case_idx, type)
        dc_each_case = acc['dc_each_case'][case_idx]
        for cls in range(len(dc_each_case)):
            dc = dc_each_case[cls]
            #print(dc)
            #logger.add_scalar(f'{type}_acc_each_case/case_{case_id:05d}/dc_{cls}', dc, epoch)

    score = (acc['dc_per_case_1'] + acc['dc_per_case_2'] + acc['dc_per_case_3']) / 3
    #score = (acc['dc_per_case_1'] )
    #print(score)
    #logger.add_scalar(f'{type}/score', score, epoch)
    return score
#
#
# def reverse_transform(vol, roi, dataset, transform):
#     min_x = max(0, roi['kidney']['min_x'] - transform.roi_error_range)
#     max_x = min(vol.shape[-1], roi['kidney']['max_x'] + transform.roi_error_range)
#     min_y = max(0, roi['kidney']['min_y'] - transform.roi_error_range)
#     max_y = min(vol.shape[-2], roi['kidney']['max_y'] + transform.roi_error_range)
#     min_z = max(0, roi['kidney']['min_z'] - dataset.roi_error_range)
#     max_z = min(roi['vol']['total_z'], roi['kidney']['max_z'] + dataset.roi_error_range)
#
#     min_height = roi['vol']['total_y']
#     min_width = roi['vol']['total_x']
#
#     roi_rows = max_y - min_y
#     roi_cols = max_x - min_x
#     max_size = max(transform.output_size[0], transform.output_size[1])
#     scale = max_size / float(max(roi_cols, roi_rows))
#     rows = int(roi_rows * scale)
#     cols = int(roi_cols * scale)
#
#     if rows < min_height:
#         h_pad_top = int((min_height - rows) / 2.0)
#         h_pad_bottom = rows + h_pad_top
#     else:
#         h_pad_top = 0
#         h_pad_bottom = min_height
#
#     if cols < min_width:
#         w_pad_left = int((min_width - cols) / 2.0)
#         w_pad_right = cols + w_pad_left
#     else:
#         w_pad_left = 0
#         w_pad_right = min_width
#
#     for i in range(len(vol)):
#         img = vol[i]
#         reverse_padding_img = img[h_pad_top:h_pad_bottom, w_pad_left:w_pad_right]
#         reverse_padding_img = reverse_padding_img.astype(np.uint8)
#         reverse_resize_img = cv2.resize(reverse_padding_img, dsize=(max_x - min_x, max_y - min_y),
#                                         interpolation=cv2.INTER_LINEAR)
#         reverse_resize_img = reverse_resize_img.astype(np.int64)
#         reverse_img = np.zeros((min_height, min_width))
#         reverse_img[min_y:max_y, min_x: max_x] = reverse_resize_img
#         vol[i] = reverse_img
#
#     size = (1, min_height, min_width)
#     vol_min_z = [np.zeros(size) for _ in range(0, min_z)]
#     vol_max_z = [np.zeros(size) for _ in range(max_z, roi['vol']['total_z'])]
#
#     vol = vol_min_z + [vol] + vol_max_z
#     vol = np.concatenate(vol, axis=0)
#
#     assert vol.shape == (roi['vol']['total_z'], roi['vol']['total_y'], roi['vol']['total_x'])
#
#     return vol
#

if __name__ == '__main__':
    main()
