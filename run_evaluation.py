import argparse
import os

# average / weighted_average / bitwise_or
IOU_MEASURE = "segm"
ENSEMBLE_METHOD = "average"
CREDIBILITY_THRESHOLD = 0.3
VIABLE_COUNTABILITY = 0
#For average
AVERAGE_ACCEPTABILITY = 0
#For weighted average
AVERAGE_ACCEPTABILITY_2 = 0.5
DEVIATION_THRESHOLD = 0.5
DEVIATION_THRESHOLD_EVAL = 0.5

#SEGMENTATION MODEL DICTIONARY
model_dict = {}

model_dict['mask_rcnn_X-101-64x4d-FPN'] = (('configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py',
                                            'checkpoints/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth',
                                            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth',
                                            0.66))
model_dict['cascade_mask_rcnn_X-101-64x4d-FPN'] = (('configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py',
                                                    'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth',
                                                    'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth',
                                                    0.58))
model_dict['hybrid_task_cascade_mask_rcnn_X-101-64x4d-FPN'] = (('configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py',
                                                                'checkpoints/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth',
                                                                'https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth',
                                                                0.56))
model_dict['gcnet_X-101-FPN_DCN_Cascade_Mask_GC(c3-c5,r4)'] = (('configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco.py',
                                                                'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth',
                                                                'https://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth',
                                                                0.6))
model_dict['scnet_X-101-64x4d-FPN'] = (('configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py',
                                        'checkpoints/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth',
                                        'https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth',
                                        0.62))
model_dict = list(model_dict.items())

test_config = 'configs/common/mstrain-poly_3x_coco_instance.py'
#test_config = 'configs/_base_/datasets/cityscapes_instance.py'
dataset_name = os.path.splitext(test_config)[0].split('/')[-1]

#AUXILIARY FUNCTIONS
import numpy as np

def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def IoU_Mask(maskA, maskB):
    intersection = np.logical_and(maskA, maskB).astype(np.uint8)
    union = np.logical_or(maskA, maskB).astype(np.uint8)
    iou = np.sum(intersection)/np.sum(union)
    return iou

#ENSEMBLE FUNCTIONS

from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from typing import List, Tuple, Union, Dict
from torch import nn
from os import path
from urllib import request
from tqdm import tqdm
import torch

import mmcv
import os.path as osp
import pycocotools.mask as mask_util

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

CITYSCAPES_CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                      'bicycle')

WORK_DIR = "work_dirs/ensemble_results/"

def inference_on_dataset(model_info):
    config, checkpoint = model_info
    
    cfg = Config.fromfile(config)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
                
    test_cfg = Config.fromfile(test_config)
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        test_cfg.data.test.test_mode = True
        samples_per_gpu = test_cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            test_cfg.data.test.pipeline = replace_ImageToTensor(
                test_cfg.data.test.pipeline)
    elif isinstance(test_cfg.data.test, list):
        for ds_cfg in test_cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in test_cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in test_cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    #rank, _ = get_dist_info()
    # allows not to create
    #mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    #timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    #json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(test_cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=test_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    #if args.fuse_conv_bn:
        #model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    #classes = model.CLASSES.copy()
    classes = list(enumerate(model.CLASSES)).copy()
    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, None, None, None)
    
    print(len(dataset))
    
    return outputs, classes, len(dataset)


def gather_results_from_model(model_name: str, model_info: Tuple[str,str], score_thr:float, person_only:bool , result_type = 'bbox'):
    if not osp.exists(WORK_DIR+model_name+"_"+dataset_name+".pkl"):
        mmcv.mkdir_or_exist(osp.abspath(WORK_DIR))
        results, original_classes, dataset_size = inference_on_dataset(model_info)
        classes = original_classes
        mmcv.dump(results, WORK_DIR+model_name+"_"+dataset_name+".pkl")
    else:
        config,checkpoint = model_info
        cfg = Config.fromfile(config)
        model = build_detector(cfg.model)
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        original_classes = list(enumerate(model.CLASSES)).copy()
        classes = original_classes
        results = mmcv.load(WORK_DIR+model_name+"_"+dataset_name+".pkl")
        dataset_size = len(results)
    if person_only:
        if len(classes) == len(COCO_CLASSES):
            classes = [(0,'person')]
        elif len(classes) == len(CITYSCAPES_CLASSES):
            classes = [(0,'person'),(1,'rider')]
    
        
    return results,classes,dataset_size

def gather_results(model_dict: Dict[str,Tuple[str,str,str]], score_thr: float, person_only: bool, result_type='bbox'):
    #model_dict = model_dict.items()
    ensemble_results = {}
    dataset_compatible = -1
    label_type = []
    for i, (name, (config,checkpoint,download_link,_)) in enumerate(model_dict):
        if not path.exists(checkpoint):
            print("Downloading",name)
            request.urlretrieve(download_link,checkpoint)
            print("Finished downloading",name)
        print("Loading inference results from model:",name)
        ensemble_results[i],classes,dataset_size = gather_results_from_model(name, (config,checkpoint), score_thr, person_only, result_type)
        label_type.append(len(classes))
        if dataset_compatible < 0 or dataset_compatible == dataset_size:
            dataset_compatible = dataset_size
        else:
            raise(Exception("Dataset sizes are not compatible"))
    return ensemble_results,classes,dataset_compatible

def group_instances(model_dict,ensemble_results, labels: List[str], dataset_size, score_thr, threshold, ensemble_method):
    #ensemble_results[model][image][bbox or segm][label][instance]
    final_results = []
    n_models = len(ensemble_results)
    #Iterate over all the images
    for img in tqdm(range(0,dataset_size)):
        bbox_group = []
        segm_group = []

        #Iterate over all the labels
        for (label_nr,label) in labels:
            bbox_results = []
            segm_results = []
            #Create a matrix of already used instances
            used_instances = []
            for cur_model in range(0,len(ensemble_results)):
                used_instances.insert(cur_model,[False]*len(ensemble_results[cur_model][img][0][label_nr]))
                
            #Iterate over all the models for a certain label and a certain image
            for cur_model in range(0,len(ensemble_results)):
                #Iterate over the current model's results on a certain label on a certain image
                for cur_instance in range(0,len(ensemble_results[cur_model][img][0][label_nr])):
                    #if not used_instances[cur_model][cur_instance] and ensemble_results[cur_model][img][0][label_nr][cur_instance][4] >= CREDIBILITY_THRESHOLD:
                    if not used_instances[cur_model][cur_instance] and ensemble_results[cur_model][img][0][label_nr][cur_instance][4] >= model_dict[cur_model][1][3]:
                        used_instances[cur_model][cur_instance] = True
                        cur_instance_group = [None for w in range(0,len(ensemble_results))]
                        cur_instance_group[cur_model] = (ensemble_results[cur_model][img][0][label_nr][cur_instance],
                                                         ensemble_results[cur_model][img][1][label_nr][cur_instance])
                        #Iterate over all the other models
                        for comp_model in range(cur_model+1,len(ensemble_results)):
                            deviations = []
                            #Iterate over each of the other model's results
                            for comp_instance in range(0,len(ensemble_results[comp_model][img][0][label_nr])):
                                if ensemble_results[comp_model][img][0][label_nr][comp_instance][4] >= model_dict[comp_model][1][3]:
                                    if not used_instances[comp_model][comp_instance]:
                                        if IOU_MEASURE == "bbox":
                                            #cur_iou = IoU(ensemble_results[cur_model][img][0][label_nr][cur_instance],ensemble_results[comp_model][img][0][label_nr][comp_instance])
                                            cur_iou = IoU_Mask(mask_util.decode(ensemble_results[cur_model][img][1][label_nr][cur_instance]),
                                                            mask_util.decode(ensemble_results[comp_model][img][1][label_nr][comp_instance]))
                                        elif IOU_MEASURE == "segm":
                                            boxA = ensemble_results[cur_model][img][0][label_nr][cur_instance]
                                            boxB = ensemble_results[comp_model][img][0][label_nr][comp_instance]
                                            xA = int(round(min(boxA[0], boxB[0])))
                                            yA = int(round(min(boxA[1], boxB[1])))
                                            xB = int(round(max(boxA[2], boxB[2])))
                                            yB = int(round(max(boxA[3], boxB[3])))
                                            cur_iou = IoU_Mask(mask_util.decode(ensemble_results[cur_model][img][1][label_nr][cur_instance])[yA:yB,xA:xB],
                                                               mask_util.decode(ensemble_results[comp_model][img][1][label_nr][comp_instance])[yA:yB,xA:xB])
                                    else:
                                        cur_iou = 0.0
                                    deviations.append(cur_iou)
                            #Check if the max iou is within the threshold and add the new instance to the group
                            if len(deviations) > 0:
                                pos = max(range(len(deviations)), key=deviations.__getitem__)
                                if deviations[pos] >= threshold:
                                    #Guarantee this instance isn't used again
                                    used_instances[comp_model][pos] = True
                                    cur_instance_group[comp_model] = (ensemble_results[comp_model][img][0][label_nr][pos],
                                                                      ensemble_results[comp_model][img][1][label_nr][pos])
                        
                        count = 0
                        for instance_i in cur_instance_group:
                            if instance_i:
                                count += 1
                                
                        # Assuming an instance group is viable if most of the networks identified it
                        if (count >= (n_models/2) + VIABLE_COUNTABILITY and not ensemble_method == "bitwise_and") or \
                           (count == n_models and ensemble_method == "bitwise_and"):
                            bbox = np.array([0.0]*5)
                            for model_result in range(0,len(cur_instance_group)):
                                if not cur_instance_group[model_result] is None:
                                    bbox = np.add(bbox,cur_instance_group[model_result][0])
                            bbox = (bbox/count)
                            confidence = bbox[4]
                            bbox = bbox.astype(int)
                            bbox[0:3] = np.around(bbox[0:3])
                            bbox_y = (bbox[3]-bbox[1]).astype(int)
                            bbox_x = (bbox[2]-bbox[0]).astype(int)
                            mask = np.zeros((bbox_y,bbox_x),dtype=int)
                            img_size = (0,0)
                            if ensemble_method == "average":
                                for model_result in range(0,len(cur_instance_group)):
                                    if not cur_instance_group[model_result] is None:
                                        decoded_mask = mask_util.decode(cur_instance_group[model_result][1])
                                        mask = mask+decoded_mask[bbox[1]:bbox[1]+bbox_y,bbox[0]:bbox[0]+bbox_x].astype(int)
                                        img_size = decoded_mask.shape
                                acceptability = max(1,count/2 + AVERAGE_ACCEPTABILITY)
                                mask = mask >= acceptability
                            elif ensemble_method == "weighted_average":
                                total_confidence = 0.0
                                for model_result in range(0,len(cur_instance_group)):
                                    if not cur_instance_group[model_result] is None:
                                        decoded_mask = mask_util.decode(cur_instance_group[model_result][1])
                                        mask = mask+(decoded_mask[bbox[1]:bbox[1]+bbox_y,bbox[0]:bbox[0]+bbox_x].astype(int) * confidence)
                                        total_confidence += confidence
                                        img_size = decoded_mask.shape
                                mask = mask >= AVERAGE_ACCEPTABILITY_2 * total_confidence
                            elif ensemble_method == "bitwise_or":
                                for model_result in range(0,len(cur_instance_group)):
                                    if not cur_instance_group[model_result] is None:
                                        decoded_mask = mask_util.decode(cur_instance_group[model_result][1])
                                        mask = mask+decoded_mask[bbox[1]:bbox[1]+bbox_y,bbox[0]:bbox[0]+bbox_x].astype(int)
                                        img_size = decoded_mask.shape
                                mask = mask > 0.0
                            elif ensemble_method == "bitwise_and":
                                for model_result in range(0,len(cur_instance_group)):
                                    decoded_mask = mask_util.decode(cur_instance_group[model_result][1])
                                    mask = mask+decoded_mask[bbox[1]:bbox[1]+bbox_y,bbox[0]:bbox[0]+bbox_x].astype(int)
                                    img_size = decoded_mask.shape
                                mask = mask == float(n_models)
                            segmentation = np.zeros(img_size).astype(bool)
                            segmentation[bbox[1]:bbox[1]+bbox_y,bbox[0]:bbox[0]+bbox_x] = mask
                            bbox = bbox.astype(float)
                            bbox[4] = confidence
                            bbox_results.append(np.array(bbox))
                            segm_results.append(mask_util.encode(np.asfortranarray(segmentation)))
                            #segm_results.append(np.array(segmentation))
            if not bbox_results is None:
                np.append(bbox_results,np.array([]))  
            bbox_group.append(np.array(bbox_results).reshape(-1,5)) 
            segm_group.append(segm_results)
        final_results.append((bbox_group,segm_group))            
                            
    return final_results

def run_ensemble(model_dict: Dict[str,Tuple[str,str,str]], score_thr: float, person_only: bool, ensemble_method: str, result_type='bbox'):
    ensemble_results,classes,dataset_size = gather_results(model_dict,score_thr,person_only,result_type)
    results = group_instances(model_dict,ensemble_results,classes,dataset_size,score_thr,DEVIATION_THRESHOLD,ensemble_method)
    #Force garbage collection in order to release memory
    return results

import pickle

def ensemble_and_evaluate(model_dict,order):
    results = run_ensemble(model_dict, CREDIBILITY_THRESHOLD,True,ENSEMBLE_METHOD,result_type='segm')

    """title = "results/"+('|'.join(str(e) for e in order))+"_e="+str(ENSEMBLE_METHOD)+"_c="+str(CREDIBILITY_THRESHOLD)+"_v="+str(VIABLE_COUNTABILITY)+"_d="+str(DEVIATION_THRESHOLD)
    if ENSEMBLE_METHOD == 'average':
        title = title + "_a="+str(AVERAGE_ACCEPTABILITY)
    elif ENSEMBLE_METHOD == 'weighted_average':
        title = title + "_a2="+str(AVERAGE_ACCEPTABILITY_2)
    title = title + ".pkl"
    with open(title, 'wb') as f:
        pickle.dump(results, f)"""

    test_cfg = Config.fromfile(test_config)
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(test_cfg.data.test, dict):
        test_cfg.data.test.test_mode = True
        samples_per_gpu = test_cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            test_cfg.data.test.pipeline = replace_ImageToTensor(
                test_cfg.data.test.pipeline)
    elif isinstance(test_cfg.data.test, list):
        for ds_cfg in test_cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in test_cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in test_cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    dataset = build_dataset(test_cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=test_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    eval_kwargs = dict(metric=["bbox","segm"],classwise=True)
    metric = dataset.evaluate(results, **eval_kwargs)
    bc_info = dataset.evaluateBC(results,DEVIATION_THRESHOLD_EVAL)
    

    if len(bc_info['tp']) > 2:
        tps = np.sum(list(bc_info['tp'].values()),axis=0)
        fps = np.sum(list(bc_info['fp'].values()),axis=0)
        tns = np.sum(list(bc_info['tn'].values()),axis=0)
        fns = np.sum(list(bc_info['fn'].values()),axis=0)

        #Copiar por aqui

        total_1 = sum([tps[0],fps[0],tns[0],fns[0]])
        total_1_small = sum([tps[1],fps[1],tns[1],fns[1]])
        total_1_medium = sum([tps[2],fps[2],tns[2],fns[2]])
        total_1_large = sum([tps[3],fps[3],tns[3],fns[3]])
        total_2 = sum([tps[4],fps[4],tns[4],fns[4]])
        total_2_small = sum([tps[5],fps[5],tns[5],fns[5]])
        total_2_medium = sum([tps[6],fps[6],tns[6],fns[6]])
        total_2_large = sum([tps[7],fps[7],tns[7],fns[7]])

        print("True Positives",tps[0],"{:.5f}".format(tps[0]/total_1),
                            tps[1],"{:.5f}".format(tps[1]/total_1_small),
                            tps[2],"{:.5f}".format(tps[2]/total_1_medium),
                            tps[3],"{:.5f}".format(tps[3]/total_1_large),
                            tps[4],"{:.5f}".format(tps[4]/total_2),
                            tps[5],"{:.5f}".format(tps[5]/total_2_small),
                            tps[6],"{:.5f}".format(tps[6]/total_2_medium),
                            tps[7],"{:.5f}".format(tps[7]/total_2_large))
        print("False Positives",fps[0],"{:.5f}".format(fps[0]/total_1),
                                fps[1],"{:.5f}".format(fps[1]/total_1_small),
                                fps[2],"{:.5f}".format(fps[2]/total_1_medium),
                                fps[3],"{:.5f}".format(fps[3]/total_1_large),
                                fps[4],"{:.5f}".format(fps[4]/total_2),
                                fps[5],"{:.5f}".format(fps[5]/total_2_small),
                                fps[6],"{:.5f}".format(fps[6]/total_2_medium),
                                fps[7],"{:.5f}".format(fps[7]/total_2_large))
        print("True Negatives",tns[0],"{:.5f}".format(tns[0]/total_1),
                            tns[1],"{:.5f}".format(tns[1]/total_1_small),
                            tns[2],"{:.5f}".format(tns[2]/total_1_medium),
                            tns[3],"{:.5f}".format(tns[3]/total_1_large),
                            tns[4],"{:.5f}".format(tns[4]/total_2),
                            tns[5],"{:.5f}".format(tns[5]/total_2_small),
                            tns[6],"{:.5f}".format(tns[6]/total_2_medium),
                            tns[7],"{:.5f}".format(tns[7]/total_2_large))
        print("False Negatives",fns[0],"{:.5f}".format(fns[0]/total_1),
                                fns[1],"{:.5f}".format(fns[1]/total_1_small),
                                fns[2],"{:.5f}".format(fns[2]/total_1_medium),
                                fns[3],"{:.5f}".format(fns[3]/total_1_large),
                                fns[4],"{:.5f}".format(fns[4]/total_2),
                                fns[5],"{:.5f}".format(fns[5]/total_2_small),
                                fns[6],"{:.5f}".format(fns[6]/total_2_medium),
                                fns[7],"{:.5f}".format(fns[7]/total_2_large))

        cg = list(bc_info['correct_guesses'].values())
        ig = list(bc_info['incorrect_guesses'].values())
        ng = list(bc_info['not_guessed'].values())
        correct_guesses = np.sum(cg,axis=0)
        incorrect_guesses = np.sum(ig,axis=0)
        not_guessed = np.sum(ng,axis=0)

        is_crowds = list(bc_info['is_crowd'].values())
        crowd_cg = list(zip(cg,is_crowds))
        crowd_cg,_ = zip(*[x for x in crowd_cg if x[1] is True])
        crowd_ig = list(zip(ig,is_crowds))
        crowd_ig,_ = zip(*[x for x in crowd_ig if x[1] is True])
        crowd_ng = list(zip(ng,is_crowds))
        crowd_ng,_ = zip(*[x for x in crowd_ng if x[1] is True])
        crowd_correct_guesses = np.sum(crowd_cg,axis=0)
        crowd_incorrect_guesses = np.sum(crowd_ig,axis=0)
        crowd_not_guessed = np.sum(crowd_ng,axis=0) 
            
        total_guesses = sum([correct_guesses[0],incorrect_guesses[0],not_guessed[0]])
        total_guesses_small = sum([correct_guesses[1],incorrect_guesses[1],not_guessed[1]])
        total_guesses_medium = sum([correct_guesses[2],incorrect_guesses[2],not_guessed[2]])
        total_guesses_large = sum([correct_guesses[3],incorrect_guesses[3],not_guessed[3]])
        total_crowd_guesses = sum([crowd_correct_guesses[0],crowd_incorrect_guesses[0],crowd_not_guessed[0]])
        total_crowd_guesses_small = sum([crowd_correct_guesses[1],crowd_incorrect_guesses[1],crowd_not_guessed[1]])
        total_crowd_guesses_medium = sum([crowd_correct_guesses[2],crowd_incorrect_guesses[2],crowd_not_guessed[2]])
        total_crowd_guesses_large = sum([crowd_correct_guesses[3],crowd_incorrect_guesses[3],crowd_not_guessed[3]])
        
        print("Correct Guesses",correct_guesses[0],"{:.5f}".format(correct_guesses[0]/total_guesses),
                                correct_guesses[1],"{:.5f}".format(correct_guesses[1]/total_guesses_small),
                                correct_guesses[2],"{:.5f}".format(correct_guesses[2]/total_guesses_medium),
                                correct_guesses[3],"{:.5f}".format(correct_guesses[3]/total_guesses_large),
                                crowd_correct_guesses[0],"{:.5f}".format(crowd_correct_guesses[0]/total_crowd_guesses),
                                crowd_correct_guesses[1],"{:.5f}".format(crowd_correct_guesses[1]/total_crowd_guesses_small),
                                crowd_correct_guesses[2],"{:.5f}".format(crowd_correct_guesses[2]/total_crowd_guesses_medium),
                                crowd_correct_guesses[3],"{:.5f}".format(crowd_correct_guesses[3]/total_crowd_guesses_large))
        print("Incorrect Guesses",incorrect_guesses[0],"{:.5f}".format(incorrect_guesses[0]/total_guesses),
                                incorrect_guesses[1],"{:.5f}".format(incorrect_guesses[1]/total_guesses_small),
                                incorrect_guesses[2],"{:.5f}".format(incorrect_guesses[2]/total_guesses_medium),
                                incorrect_guesses[3],"{:.5f}".format(incorrect_guesses[3]/total_guesses_large),
                                crowd_incorrect_guesses[0],"{:.5f}".format(crowd_incorrect_guesses[0]/total_crowd_guesses),
                                crowd_incorrect_guesses[1],"{:.5f}".format(crowd_incorrect_guesses[1]/total_crowd_guesses_small),
                                crowd_incorrect_guesses[2],"{:.5f}".format(crowd_incorrect_guesses[2]/total_crowd_guesses_medium),
                                crowd_incorrect_guesses[3],"{:.5f}".format(crowd_incorrect_guesses[3]/total_crowd_guesses_large))
        print("Not Guessed",not_guessed[0],"{:.5f}".format(not_guessed[0]/total_guesses),
                            not_guessed[1],"{:.5f}".format(not_guessed[1]/total_guesses_small),
                            not_guessed[2],"{:.5f}".format(not_guessed[2]/total_guesses_medium),
                            not_guessed[3],"{:.5f}".format(not_guessed[3]/total_guesses_large),
                            crowd_not_guessed[0],"{:.5f}".format(crowd_not_guessed[0]/total_crowd_guesses),
                            crowd_not_guessed[1],"{:.5f}".format(crowd_not_guessed[1]/total_crowd_guesses_small),
                            crowd_not_guessed[2],"{:.5f}".format(crowd_not_guessed[2]/total_crowd_guesses_medium),
                            crowd_not_guessed[3],"{:.5f}".format(crowd_not_guessed[3]/total_crowd_guesses_large))

def ordering_recursion(models, missing_iterations, used_array, order_array):
    if missing_iterations == 0:
        ordered_model_dict = []
        print("Model Order: ",end="")
        for i in range(0,len(order_array)):
            ordered_model_dict.append(models[order_array[i]])
            print(ordered_model_dict[i][0],end=" -> ")
        print("")
        ensemble_and_evaluate(ordered_model_dict,order_array)
    else:
        for i in range(0, len(models)):
            if not used_array[i]:
                cur_used_array = used_array.copy()
                cur_used_array[i] = True
                cur_order_array = order_array.copy()
                cur_order_array.append(i)
                ordering_recursion(models,missing_iterations-1,cur_used_array,cur_order_array)

def non_ordered_recursion(models, next_model, missing_iterations, used_array, order_array):       
    if missing_iterations == 0:
        ordered_model_dict = []
        print("Model Order: ",end="")
        for i in range(0,len(order_array)):
            ordered_model_dict.append(models[order_array[i]])
            print(ordered_model_dict[i][0],end=" -> ")
        print("")
        ensemble_and_evaluate(ordered_model_dict,order_array)
    else:
        for i in range(next_model, len(models)):
            if not used_array[i]:
                cur_used_array = used_array.copy()
                cur_used_array[i] = True
                cur_order_array = order_array.copy()
                cur_order_array.append(i)
                non_ordered_recursion(models,i,missing_iterations-1,cur_used_array,cur_order_array)

def ensemble_permutations(models, ordered):
    used_array = [False] * len(models)
    for i in range(1,len(models)+1): #[1]:
        if ordered:
            ordering_recursion(models, i, used_array, [])
        else:
            non_ordered_recursion(models,0, i, used_array, [])

def main():
    parser = argparse.ArgumentParser(description='Run the ensemble')
    parser.add_argument('ordered', default='False', 
                        help='True or False depending if taking order into account')
    parser.add_argument('-data','--dataset', default='coco',
                        help='Dataset: coco/cityscapes')
    parser.add_argument('-e','--ensemble_method', default='average',
                        help='Ensemble Method: average/weighted_average/bitwise_or')
    parser.add_argument('-c','--credibility_threshold', default=0.3,
                        help='Credibility Threshold: Between 0 and 1')
    parser.add_argument('-v','--viable_countability', default=0,
                        help='Viable Countability: Affects the amount of instances required to form an instance group')
    parser.add_argument('-a','--average_acceptability', default=0,
                        help='Average Acceptability: Affects the amount of confidence required for a pixel to make it into the final mask in average')
    parser.add_argument('-a2','--average_acceptability_2', default=0.5,
                        help='Average Acceptability 2: Affects the amount of confidence required for a pixel to make it into the final mask in weighted average (Between 0 and 1)')
    parser.add_argument('-d','--deviation_threshold', default=0.5,
                        help='Deviation Threshold: Defines the minimum amount of IoU in order for an instance to be part of another')

    args = parser.parse_args()


    try:
        if args.ordered == 'True':
            ordered = True
        elif args.ordered == 'False':
            ordered = False
        else:
            raise(TypeError('\'ordered\' must be \'True\' of \'False\''))

        if args.dataset:
            global test_config
            global dataset_name
            if args.dataset == "coco":
                test_config = 'configs/common/mstrain-poly_3x_coco_instance.py'
            elif args.dataset == "cityscapes":
                test_config = 'configs/_base_/datasets/cityscapes_instance.py'
            else:
                print("Unknown Dataset")
            dataset_name = os.path.splitext(test_config)[0].split('/')[-1]

        if args.ensemble_method:
            if args.ensemble_method in ['average','weighted_average','bitwise_or','bitwise_and']:
                global ENSEMBLE_METHOD
                ENSEMBLE_METHOD = args.ensemble_method
            else:
                raise(TypeError('\'ensemble_method\' must be either \'average\' \'weighted_average\' or \'bitwise_or\''))

        if args.credibility_threshold:
            if 0 <= float(args.credibility_threshold) <= 1:
                global CREDIBILITY_THRESHOLD
                CREDIBILITY_THRESHOLD = float(args.credibility_threshold)
            else:
                raise(TypeError('\'credibility_threshold\' must be between 0 or 1'))
        
        if args.viable_countability:
            global VIABLE_COUNTABILITY
            VIABLE_COUNTABILITY = int(args.viable_countability)

        if args.average_acceptability:
            global AVERAGE_ACCEPTABILITY
            AVERAGE_ACCEPTABILITY = int(args.average_acceptability)

        if args.average_acceptability_2:
            if 0 <= float(args.average_acceptability_2) <= 1:
                global AVERAGE_ACCEPTABILITY_2
                AVERAGE_ACCEPTABILITY_2 = float(args.average_acceptability_2)
            else:
                raise(TypeError('\'average_acceptability_2\' must be between 0 or 1'))

        if args.deviation_threshold:
            if 0 <= float(args.deviation_threshold) <= 1:
                global DEVIATION_THRESHOLD
                DEVIATION_THRESHOLD = float(args.deviation_threshold)
            else:
                raise(TypeError('\'deviation_threshold\' must be between 0 or 1'))       
    except TypeError as err:
        print(err)
    

    print("Ordered",ordered)
    print("Dataset",dataset_name)
    print("Ensemble Method",ENSEMBLE_METHOD)
    print("Credibility Threshold",CREDIBILITY_THRESHOLD)
    print("Viable Countability",VIABLE_COUNTABILITY)
    print("Average Acceptability",AVERAGE_ACCEPTABILITY)
    print("Average Acceptability 2",AVERAGE_ACCEPTABILITY_2)
    print("Deviation Threshold",DEVIATION_THRESHOLD)

    perm_results = ensemble_permutations(model_dict,ordered)

if __name__ == "__main__":
    main()