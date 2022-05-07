_base_ = [
    'mask_rcnn_r50_fpn.py',
    'filtered_coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 'default_runtime.py'
]
