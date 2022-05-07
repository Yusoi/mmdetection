import re
import argparse
import os

"""
model_dict = {}

#Mask R-CNN
model_dict['mask_rcnn_X-101-64x4d-FPN'] = (('configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py',
                                            'checkpoints/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth',
                                            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth'))
#Cascade Mask R-CNN
model_dict['cascade_mask_rcnn_X-101-64x4d-FPN'] = (('configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py',
                                                    'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth',
                                                    'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth'))
#HTC
model_dict['hybrid_task_cascade_mask_rcnn_X-101-64x4d-FPN'] = (('configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py',
                                                                'checkpoints/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth',
                                                                'https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth'))
#GCNet
model_dict['gcnet_X-101-FPN_DCN_Cascade_Mask_GC(c3-c5,r4)'] = (('configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco.py',
                                                                'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth',
                                                                'https://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth'))
#SCNet
model_dict['scnet_X-101-64x4d-FPN'] = (('configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py',
                                        'checkpoints/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth',
                                        'https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth'))
#Carafe
model_dict['mask_rcnn_r50_fpn_carafe_1x_coco'] = (('configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py',
                                                   'checkpoints/mask_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.393__segm_mAP-0.358_20200503_135957-8687f195.pth',
                                                   'https://download.openmmlab.com/mmdetection/v2.0/carafe/mask_rcnn_r50_fpn_carafe_1x_coco/mask_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.393__segm_mAP-0.358_20200503_135957-8687f195.pth'))
#Deformable Convolution
model_dict['cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco'] = (('configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py',
                                                                       'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth',
                                                                       'https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth'))
#Group Normalization
model_dict['mask_rcnn_r101_fpn_gn-all_3x_coco'] = (('configs/gn/mask_rcnn_r101_fpn_gn-all_3x_coco.py',
                                                    'checkpoints/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609-0df864f4.pth',
                                                    'https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_3x_coco/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609-0df864f4.pth'))
#Group Normalization + Weight Standardization
model_dict['mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco'] = (('configs/gn+ws/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco.py',
                                                              'checkpoints/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco_20200213-57b5a50f.pth',
                                                              'https://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco_20200213-57b5a50f.pth'))
#Detectors
model_dict['detectors_htc_r101_20e_coco'] = (('configs/detectors/detectors_htc_r101_20e_coco.py',
                                              'checkpoints/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth',
                                              'https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r101_20e_coco/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'))

model_dict = list(model_dict.items())"""

model_dict = []
"""
model_dict.append(('hybrid_task_cascade_mask_rcnn_X-101-64x4d-FPN',('configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py',
                                                                'checkpoints/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth',
                                                                'https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth')))
model_dict.append(('detectors_htc_r101_20e_coco',('configs/detectors/detectors_htc_r101_20e_coco.py',
                                                  'checkpoints/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth',
                                                  'https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r101_20e_coco/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth')))
model_dict.append(('cascade_mask_rcnn_X-101-64x4d-FPN',('configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py',
                                                        'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth',
                                                        'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth')))
model_dict.append(('cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco',('configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py',
                                                                           'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth',
                                                                           'https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth')))
model_dict.append(('gcnet_X-101-FPN_DCN_Cascade_Mask_GC(c3-c5,r4)',('configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco.py',
                                                                    'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth',
                                                                    'https://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth')))"""

model_dict.append(('mask_rcnn_r50_fpn_1x_coco.py',('configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
                                                   'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth',
                                                   'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth')))

model_dict.append(('mask_rcnn_r50_fpn_1x_cityscapes',('configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py',
                                                      'checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth',
                                                      'https://download.openmmlab.com/mmdetection/v2.0/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth')))


key_list = [model[0] for model in model_dict]

import glob
import os

def main():
    parser = argparse.ArgumentParser(description="Parse result files to obtain the highest value of the mask AP")
    parser.add_argument('folder', help="File to resolve")
    args = parser.parse_args()

    for file in glob.glob(args.folder+"/results/*.txt"):
        result_file = open(file,"r").read()
        model_names = re.findall(r"^Model Order: ([^\n]+)",result_file,flags=re.MULTILINE)
        tps = re.findall(r"^TP (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        fps = re.findall(r"^FP (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        tns = re.findall(r"^TN (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        fns = re.findall(r"^FN (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        oa = re.findall(r"^OA (.+) (.+)",result_file,flags=re.MULTILINE)
        p = re.findall(r"^P (.+) (.+)",result_file,flags=re.MULTILINE)
        r = re.findall(r"^R (.+) (.+)",result_file,flags=re.MULTILINE)
        f1 = re.findall(r"^F1 (.+) (.+)",result_file,flags=re.MULTILINE)
        iou = re.findall(r"^IoU (.+) (.+)",result_file,flags=re.MULTILINE)
        cg = re.findall(r"^CG (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        ng = re.findall(r"^NG (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        tcg = re.findall(r"^TCG (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        tng = re.findall(r"^TNG (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        tig = re.findall(r"^TIG (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        ugr11 = re.findall(r"^UGR11 (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        ugr31 = re.findall(r"^UGR31 (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)
        ugr13 = re.findall(r"^UGR13 (.+) (.+) (.+) (.+)",result_file,flags=re.MULTILINE)

        with open("transfer_learning/csv/"+os.path.splitext(file)[0].split("/")[-1]+".csv","w") as f:
            for model in key_list:
                f.write(model+";")
            f.write("tp%;tp_s%;tp_m%;tp_l%;fp%;fp_s%;fp_m%;fp_l%;tn%;tn_s%;tn_m%;tn_l%;fn%;fn_s%;fn_m%;fn_l%;")
            f.write("oa_full;oa_correct;p_full;p_correct;r_full;r_correct;f1_full;f1_correct;iou_full;iou_correct;")
            f.write("cg%;cg_s%;cg_m%;cg_l%;ng%;ng_s%;ng_m%;ng_l%;tcg%;tcg_s%;tcg_m%;tcg_l%;tng%;tng_s%;tng_m%;tng_l%;tig%;tig_s%;tig_m%;tig_l%;")
            f.write("ugr11%;ugr11_s%;ugr11_m%;ugr11_l%;ugr31%;ugr31_s%;ugr31_m%;ugr31_l%;ugr13%;ugr13_s%;ugr13_m%;ugr13_l%\n")

            for i in range(0,len(model_names)):
                cur_array = [0 for i in range(0,len(key_list))]
                cur_model = 1
                for j in model_names[i].split(" -> "):
                    for k in range(0,len(key_list)):
                        if j == key_list[k]:
                            cur_array[k] = cur_model
                            cur_model += 1
                            break
                for model in cur_array:
                    f.write(str(model)+";")
                
                f.write(tps[i][0]+";"+tps[i][1]+";"+tps[i][2]+";"+tps[i][3]+";")
                f.write(fps[i][0]+";"+fps[i][1]+";"+fps[i][2]+";"+fps[i][3]+";")
                f.write(tns[i][0]+";"+tns[i][1]+";"+tns[i][2]+";"+tns[i][3]+";")
                f.write(fns[i][0]+";"+fns[i][1]+";"+fns[i][2]+";"+fns[i][3]+";")
                f.write(oa[i][0]+";"+oa[i][1]+";")
                f.write(p[i][0]+";"+p[i][1]+";")
                f.write(r[i][0]+";"+r[i][1]+";")
                f.write(f1[i][0]+";"+f1[i][1]+";")
                f.write(iou[i][0]+";"+iou[i][1]+";")
                f.write(cg[i][0]+";"+cg[i][1]+";"+cg[i][2]+";"+cg[i][3]+";")
                f.write(ng[i][0]+";"+ng[i][1]+";"+ng[i][2]+";"+ng[i][3]+";")  
                f.write(tcg[i][0]+";"+tcg[i][1]+";"+tcg[i][2]+";"+tcg[i][3]+";")
                f.write(tng[i][0]+";"+tng[i][1]+";"+tng[i][2]+";"+tng[i][3]+";") 
                f.write(tig[i][0]+";"+tig[i][1]+";"+tig[i][2]+";"+tig[i][3]+";") 
                f.write(ugr11[i][0]+";"+ugr11[i][1]+";"+ugr11[i][2]+";"+ugr11[i][3]+";")
                f.write(ugr31[i][0]+";"+ugr31[i][1]+";"+ugr31[i][2]+";"+ugr31[i][3]+";") 
                f.write(ugr13[i][0]+";"+ugr13[i][1]+";"+ugr13[i][2]+";"+ugr13[i][3]) 
                f.write("\n")
        
if __name__ == '__main__':
    main()