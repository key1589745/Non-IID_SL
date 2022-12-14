# -*- coding:utf-8 -*-

import torch
import SimpleITK as sitk
import numpy as np


def dice_compute(pred, groundtruth):           #batchsize*channel*W*W

    dice=[]
    for i in range(4):
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]


    return np.array(dice,dtype=np.float32)




def IOU_compute(pred, groundtruth):
    iou=[]
    for i in range(4):
        iou_i = (np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)-np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)
        iou=iou+[iou_i]


    return np.array(iou,dtype=np.float32)


def Hausdorff_compute(pred,groundtruth,num_class,spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1,num_class, 5))
    surface_distance_results = np.zeros((1,num_class, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(num_class):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()

            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)


    return overlap_results,surface_distance_results



class Evaluator(object):
    def __init__(self,data_loader_vali,num_cls):

        self.vali_loaders = data_loader_vali
        self.num_cls = num_cls

    def eval(self, model,client):

        total_overlap = np.zeros((1, self.num_cls, 5))
        res = {}
        model.eval()
        for vali_batch in self.vali_loaders[client]:

            imgs = torch.from_numpy(vali_batch['data']).cuda(non_blocking=True)
            labs = vali_batch['seg']

            output= model(imgs)

            truemax, truearg0 = torch.max(output, 1, keepdim=False)

            truearg = truearg0.detach().cpu().numpy().astype(np.uint8)

            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            overlap_result, _ = Hausdorff_compute(truearg, labs, self.num_cls, (1.5,1.5,10,1))

            total_overlap = np.concatenate((total_overlap, overlap_result), axis=0)

            #del input, truearg0, truemax

        meanDice = np.round(np.mean(total_overlap[1:,:,1], axis=0),4)
        res = dict(zip(['Myo','LV','RV'],meanDice[1:]))

        return res


class knowledge_metric(Evaluator):
    # evaluate the knowledge forgetting during local training
    def eval(self, model,client,labels):

        total_overlap = np.zeros((1, self.num_cls, 5))
        res = {}
        model.eval()
        for vali_batch in self.vali_loaders[client]:

            imgs = torch.from_numpy(vali_batch['data']).cuda(non_blocking=True)
            labs = vali_batch['seg']

            output= model(imgs)

            truemax, truearg0 = torch.max(output, 1, keepdim=False)

            truearg = truearg0.detach().cpu().numpy().astype(np.uint8)

            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            overlap_result, _ = Hausdorff_compute(truearg, labs, self.num_cls, (1.5,1.5,10,1))

            total_overlap = np.concatenate((total_overlap, overlap_result), axis=0)


        return np.mean(total_overlap[1:,labels,1],1)
