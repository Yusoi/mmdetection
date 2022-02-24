# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset

# CUSTOM STUFF #
    
import numpy as np
import datetime
import time
from collections import defaultdict
import pycocotools.mask as mask_util
from pycocotools import mask as maskUtils
import copy
from pycocotools.cocoeval import Params

@DATASETS.register_module()
class CocoDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            #print(ann_ids)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break
                
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
    
    
    def IoU(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def IoU_Mask(self, maskA, maskB):
        intersection = np.logical_and(maskA, maskB).astype(np.uint8)
        union = np.logical_or(maskA, maskB).astype(np.uint8)
        iou = np.sum(intersection)/np.sum(union)
        return iou
    
    def match_instances(self,
                        gt_segm,
                        dt_segm,
                        threshold):
        IoU = self.IoU
        IoU_Mask = self.IoU_Mask
        matched = []
        gt_unmatched = []
        dt_unmatched = []
        used = [False for i in range(0,len(dt_segm))]
        if len(gt_segm) == 0 or len(dt_segm) == 0:
            matched = []
            gt_unmatched = gt_segm
            dt_unmatched = dt_segm
            return matched, gt_unmatched, dt_unmatched
        
        crowds = []
        for gt in range(0,len(gt_segm)):
            if gt_segm[gt]['iscrowd'] == 1:
                crowds.append(gt)
            else:
                gt_bbox = gt_segm[gt]['bbox'].copy()
                gt_bbox[2] = gt_bbox[0]+gt_bbox[2]
                gt_bbox[3] = gt_bbox[1]+gt_bbox[3]
                gt_segm_decoded = mask_util.decode(gt_segm[gt]['segmentation'])
                deviations = []
                for dt in range(0,len(dt_segm)):
                    if not used[dt]:
                        dt_bbox = dt_segm[dt]['bbox'].copy()
                        dt_bbox[2] = dt_bbox[0]+dt_bbox[2]
                        dt_bbox[3] = dt_bbox[1]+dt_bbox[3]
                        #cur_iou = IoU(gt_bbox,dt_bbox)

                        xA = int(round(min(gt_bbox[0], dt_bbox[0])))
                        yA = int(round(min(gt_bbox[1], dt_bbox[1])))
                        xB = int(round(max(gt_bbox[2], dt_bbox[2])))
                        yB = int(round(max(gt_bbox[3], dt_bbox[3])))
                        cur_iou = IoU_Mask(gt_segm_decoded[yA:yB,xA:xB],
                                           mask_util.decode(dt_segm[dt]['segmentation'])[yA:yB,xA:xB])
                        deviations.append(cur_iou)
                    else:
                        deviations.append(0.0)
                pos = max(range(len(deviations)), key=deviations.__getitem__)
                if deviations[pos] >= threshold:
                    #Guarantee this instance isn't used again
                    used[pos] = True
                    matched.append((gt_segm[gt],dt_segm[pos],deviations[pos]))
                else:
                    gt_unmatched.append(gt_segm[gt])
                    
        for i in range(0,len(used)):
            if not used[i]:
                dt_segm_decoded = mask_util.decode(dt_segm[i]['segmentation'])
                dt_bbox = dt_segm[dt]['bbox'].copy()
                dt_bbox[2] = dt_bbox[0]+dt_bbox[2]
                dt_bbox[3] = dt_bbox[1]+dt_bbox[3]
                xA,xB,yA,yB = int(round(dt_bbox[0])),int(round(dt_bbox[2])),int(round(dt_bbox[1])),int(round(dt_bbox[3]))
                deviations = []
                for c in crowds:
                    intersection = np.logical_and(mask_util.decode(gt_segm[c]['segmentation']),
                                                  dt_segm_decoded)
                    cur_iou = IoU_Mask(dt_segm_decoded[yA:yB,xA:xB],
                                       intersection[yA:yB,xA:xB])
                    deviations.append(cur_iou)
                if len(crowds) != 0:
                    pos = max(range(len(deviations)), key=deviations.__getitem__)
                    if deviations[pos] < threshold:
                        dt_unmatched.append(dt_segm[i])
                else:
                    dt_unmatched.append(dt_segm[i])
            
        return matched, gt_unmatched, dt_unmatched
    
    
    
    def custom_evaluate(self,
                   results,
                   threshold,
                   is_cityscapes=False):
        result_files, tmp_dir = self.format_results(results, None)
        eval_results = OrderedDict()
        cocoGt = self.coco
        imgIds = sorted(cocoGt.getImgIds())
        catIds = sorted(cocoGt.getCatIds())
        imgIds = list(np.unique(imgIds))
        catIds = list(np.unique(catIds))
        
        
        metrics = ['segm']
        iou_type = 'segm'
        for metric in metrics:
            #iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                imgId = 0
                tp, fp, tn, fn, oa, p, r, f1, cg, ig, ng, iou  = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
                
                cur_tp = [[0],[0],[0],[0]]
                cur_fp = [[0],[0],[0],[0]]
                cur_tn = [[0],[0],[0],[0]]
                cur_fn = [[0],[0],[0],[0]]
                cur_oa = [[0],[0]]
                cur_p = [[0],[0]]
                cur_r = [[0],[0]]
                cur_f1 = [[0],[0]]
                cur_iou = [[0],[0]]
                cur_cg = [0,0,0,0]
                cur_ig = [0,0,0,0]
                cur_ng = [0,0,0,0]

                return {
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn,
                    'oa': oa,
                    'p': p,
                    'r': r,
                    "f1": f1,
                    'iou': iou,
                    'cg': cg,
                    'ig': ig,
                    'ng': ng,
                }           
        
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        gts=cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIds, catIds=catIds))
        dts=cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=imgIds, catIds=catIds))

        # convert ground truth to mask if iouType == 'segm'
        if iou_type == 'segm':
            _toMask(gts, cocoGt)
            _toMask(dts, cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        _gts = defaultdict(list)       # gt for evaluation
        _dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            _gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            _dts[dt['image_id'], dt['category_id']].append(dt)
        
        tp, fp, tn, fn, oa, p, r, f1, cg, ig, ng, iou  = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        match_instances = self.match_instances

        if not is_cityscapes:
            cats = [1]
        else:
            cats = [24,25]
        
        catdict = {}
        for catId in catIds:
            catdict[catId] = 0
        for imgId in imgIds:
            for catId in catIds:
                catdict[catId] += len(_gts[imgId,catId])
        
        for imgId in imgIds:
            # M2 / M2 Small / M2 Medium / M2 Large
            cur_tp = [[],[],[],[]]
            cur_fp = [[],[],[],[]]
            cur_tn = [[],[],[],[]]
            cur_fn = [[],[],[],[]]
            # Method 2 All / Method 2 Correct
            cur_oa = [[],[]]
            cur_p = [[],[]]
            cur_r = [[],[]]
            cur_f1 = [[],[]]
            cur_iou = [[],[]]
            cur_cg = [0,0,0,0]
            cur_ig = [0,0,0,0]
            cur_ng = [0,0,0,0]
            
             
            gt_segm = []
            dt_segm = []
            
            for catId in cats:
                if (imgId,catId) in _gts:
                    gt_segm = gt_segm + _gts[imgId,catId]
                if (imgId,catId) in _dts:
                    dt_segm = dt_segm + _dts[imgId,catId]         

            matched, gt_unmatched, dt_unmatched = match_instances(gt_segm, dt_segm, threshold) 

            for match in matched:
                gt_bbox = match[0]['bbox']
                dt_bbox = match[1]['bbox']
                gt_bbox = np.around(gt_bbox).astype(int)
                dt_bbox = np.around(dt_bbox).astype(int)           
                gt_area = gt_bbox[2] * gt_bbox[3]
                gt_bbox[2] = gt_bbox[0]+gt_bbox[2]
                gt_bbox[3] = gt_bbox[1]+gt_bbox[3]
                dt_bbox[2] = dt_bbox[0]+dt_bbox[2]
                dt_bbox[3] = dt_bbox[1]+dt_bbox[3]

                gt_counts = mask_util.decode(match[0]['segmentation'])
                dt_counts = mask_util.decode(match[1]['segmentation'])

                """# METHOD 1

                gt_mask = gt_counts[gt_bbox[1]:gt_bbox[3],gt_bbox[0]:gt_bbox[2]]
                dt_mask = dt_counts[gt_bbox[1]:gt_bbox[3],gt_bbox[0]:gt_bbox[2]]

                temp_tp_1 = np.count_nonzero(np.logical_and(gt_mask,dt_mask))
                temp_fp_1 = np.count_nonzero(np.logical_and(np.logical_not(gt_mask),dt_mask))
                temp_tn_1 = np.count_nonzero(np.logical_and(np.logical_not(gt_mask),np.logical_not(dt_mask)))
                temp_fn_1 = np.count_nonzero(np.logical_and(gt_mask,np.logical_not(dt_mask)))
                total_pixels_1 = temp_tp_1+temp_fp_1+temp_tn_1+temp_fn_1
                temp_p1 = temp_tp_1/(temp_tp_1+temp_fp_1)
                temp_r1 = temp_tp_1/(temp_tp_1+temp_fn_1)

                cur_tp[0].append(temp_tp_1/total_pixels_1)
                cur_fp[0].append(temp_fp_1/total_pixels_1)
                cur_tn[0].append(temp_tn_1/total_pixels_1)
                cur_fn[0].append(temp_fn_1/total_pixels_1)
                cur_p[0].append(temp_p1)
                cur_p[1].append(temp_p1)
                cur_r[0].append(temp_r1)
                cur_r[1].append(temp_r1)
                cur_f1[0].append(2*temp_r1*temp_p1/(temp_r1+temp_p1))
                cur_f1[1].append(2*temp_r1*temp_p1/(temp_r1+temp_p1))

                # Small
                if gt_area < 32^9:
                    area_i = 1
                # Medium    
                elif gt_area <= 96^2:
                    area_i = 2
                # Large
                else:
                    area_i = 3

                cur_cg[0] += 1    
                cur_cg[area_i] += 1    

                cur_tp[area_i].append(temp_tp_1/total_pixels_1)
                cur_fp[area_i].append(temp_fp_1/total_pixels_1)
                cur_tn[area_i].append(temp_tn_1/total_pixels_1)
                cur_fn[area_i].append(temp_fn_1/total_pixels_1)"""

                # METHOD 2

                union_bbox = np.around([min(gt_bbox[0],dt_bbox[0]),min(gt_bbox[1],dt_bbox[1]),max(gt_bbox[2],dt_bbox[2]),max(gt_bbox[3],dt_bbox[3])])
                union_area = (union_bbox[3]-union_bbox[1])*(union_bbox[2]-union_bbox[0])
                gt_mask_union = gt_counts[union_bbox[1]:union_bbox[3],union_bbox[0]:union_bbox[2]]
                dt_mask_union = dt_counts[union_bbox[1]:union_bbox[3],union_bbox[0]:union_bbox[2]]

                temp_tp_2 = np.count_nonzero(np.logical_and(gt_mask_union,dt_mask_union))
                temp_fp_2 = np.count_nonzero(np.logical_and(np.logical_not(gt_mask_union),dt_mask_union))
                temp_tn_2 = np.count_nonzero(np.logical_and(np.logical_not(gt_mask_union),np.logical_not(dt_mask_union)))
                temp_fn_2 = np.count_nonzero(np.logical_and(gt_mask_union,np.logical_not(dt_mask_union)))
                total_pixels_2 = temp_tp_2+temp_fp_2+temp_tn_2+temp_fn_2
                temp_oa2 = (temp_tp_2+temp_tn_2)/(temp_tp_2+temp_tn_2+temp_fp_2+temp_fn_2)
                temp_p2 = temp_tp_2/(temp_tp_2+temp_fp_2)
                temp_r2 = temp_tp_2/(temp_tp_2+temp_fn_2)

                cur_tp[0].append(temp_tp_2/total_pixels_2)
                cur_fp[0].append(temp_fp_2/total_pixels_2)
                cur_tn[0].append(temp_tn_2/total_pixels_2)
                cur_fn[0].append(temp_fn_2/total_pixels_2)
                cur_oa[0].append(temp_oa2)
                cur_oa[1].append(temp_oa2)
                cur_p[0].append(temp_p2)
                cur_p[1].append(temp_p2)
                cur_r[0].append(temp_r2)
                cur_r[1].append(temp_r2)
                cur_f1[0].append(2*temp_r2*temp_p2/(temp_r2+temp_p2))
                cur_f1[1].append(2*temp_r2*temp_p2/(temp_r2+temp_p2))
                cur_iou[0].append(match[2])
                cur_iou[1].append(match[2])

                # Small
                if union_area < 32^9:
                    area_j = 1
                # Medium    
                elif union_area <= 96^2:
                    area_j = 2
                # Large
                else:
                    area_j = 3
                    
                cur_cg[0] += 1    
                cur_cg[area_j] += 1   

                cur_tp[area_j].append(temp_tp_2/total_pixels_2)
                cur_fp[area_j].append(temp_fp_2/total_pixels_2)
                cur_tn[area_j].append(temp_tn_2/total_pixels_2)
                cur_fn[area_j].append(temp_fn_2/total_pixels_2)

            for img in dt_unmatched:
                cur_bbox = img['bbox']
                cur_bbox = np.around(cur_bbox).astype(int)
                cur_area = cur_bbox[2] * cur_bbox[3]

                # Small
                if cur_area < 32^9:
                    area_k = 1
                # Medium    
                elif cur_area <= 96^2:
                    area_k = 2
                # Large
                else:
                    area_k = 3

                cur_ig[0] += 1    
                cur_ig[area_k] += 1 
                
                cur_oa[0].append(0.0)
                cur_p[0].append(0.0)
                cur_r[0].append(0.0)
                cur_f1[0].append(0.0)
                cur_iou[0].append(0.0)

            for img in gt_unmatched:
                cur_bbox = img['bbox']
                cur_bbox = np.around(cur_bbox).astype(int)
                cur_area = cur_bbox[2] * cur_bbox[3]

                # Small
                if cur_area < 32^9:
                    area_w = 1
                # Medium    
                elif cur_area <= 64^2:
                    area_w = 2
                # Large
                else:
                    area_w = 3

                cur_ng[0] += 1
                cur_ng[area_w] += 1 
                
                cur_oa[0].append(0.0)
                cur_p[0].append(0.0)
                cur_r[0].append(0.0)
                cur_f1[0].append(0.0)
                cur_iou[0].append(0.0)
                
            tp[imgId] = cur_tp.copy()
            fp[imgId] = cur_fp.copy()
            tn[imgId] = cur_tn.copy()
            fn[imgId] = cur_fn.copy()
            cg[imgId] = cur_cg.copy()
            ig[imgId] = cur_ig.copy()
            ng[imgId] = cur_ng.copy()
            oa[imgId] = cur_oa.copy()
            p[imgId] = cur_p.copy()
            r[imgId] = cur_r.copy()
            f1[imgId] = cur_f1.copy()
            iou[imgId] = cur_iou.copy()
                
        return {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'oa': oa,
            'p': p,
            'r': r,
            "f1": f1,
            'iou': iou,
            'cg': cg,
            'ig': ig,
            'ng': ng,
        }     
    
    def gt_return(self):
        eval_results = OrderedDict()
        cocoGt = self.coco
        imgIds = sorted(cocoGt.getImgIds())
        catIds = sorted(cocoGt.getCatIds())
        imgIds = list(np.unique(imgIds))
        catIds = list(np.unique(catIds))
        
        metrics = ['segm']
        iou_type = 'segm'
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        gts=cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIds, catIds=catIds))

        # convert ground truth to mask if iouType == 'segm'
        if iou_type == 'segm':
            _toMask(gts, cocoGt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        _gts = defaultdict(list)       # gt for evaluation    
        for gt in gts:
            if gt['category_id'] == 1:
                _gts[gt['image_id']].append(gt)
        
        return _gts.copy()
                    
    
    '''def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = '.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results'''
