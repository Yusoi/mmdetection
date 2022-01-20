import re
import argparse
import os

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

model_dict = list(model_dict.items())
key_list = [model[0] for model in model_dict]

import glob
import os

def main():
    parser = argparse.ArgumentParser(description="Parse result files to obtain the highest value of the mask AP")
    parser.add_argument('folder', help="File to resolve")
    args = parser.parse_args()

    for file in glob.glob(args.folder+"/*.txt"):
        result_file = open(file,"r").read()
        model_names = re.findall(r"Model Order: ([^\n]+)",result_file)
        tps = re.findall(r"TP (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        fps = re.findall(r"FP (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        tns = re.findall(r"TN (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        fns = re.findall(r"FN (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        p = re.findall(r"P (.+) (.+) (.+) (.+)",result_file)
        r = re.findall(r"R (.+) (.+) (.+) (.+)",result_file)
        f1 = re.findall(r"F1 (.+) (.+) (.+) (.+)",result_file)
        cg = re.findall(r"CG (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        ig = re.findall(r"IG (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        ng = re.findall(r"NG (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)

        with open("csv_results_coco/"+os.path.splitext(file)[0].split("/")[-1]+".csv","w") as f:
            for model in key_list:
                f.write(model+";")
            f.write("tp%;tp_s%;tp_m%;tp_l%;tp_2%;tp_2_s%;tp_2_m%;tp_2_l%;fp%;fp_s%;fp_m%;fp_l%;fp_2%;fp_2_s%;fp_2_m%;fp_2_l%;tn%;tn_s%;tn_m%;tn_l%;tn_2%;tn_2_s%;tn_2_m%;tn_2_l%;fn%;fn_s%;fn_m%;fn_l%;fn_2%;fn_2_s%;fn_2_m%;fn_2_l%;p_1_full;p_1_correct;p_2_full;p_2_correct;r_1_full;r_1_correct;r_2_full;r_2_correct;f1_1_full;f1_1_correct;f1_2_full;f1_2_correct;cg;cg%;cg_s;cg_s%;cg_m;cg_m%;cg_l;cg_l%;ig;ig%;ig_s;ig_s%;ig_m;ig_m%;ig_l;ig_l%;ng;ng%;ng_s;ng_s%;ng_m;ng_m%;ng_l;ng_l%\n")

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
                
                f.write(tps[i][0]+";"+tps[i][1]+";"+tps[i][2]+";"+tps[i][3]+";"+
                        tps[i][4]+";"+tps[i][5]+";"+tps[i][6]+";"+tps[i][7]+";")
                f.write(fps[i][0]+";"+fps[i][1]+";"+fps[i][2]+";"+fps[i][3]+";"+
                        fps[i][4]+";"+fps[i][5]+";"+fps[i][6]+";"+fps[i][7]+";")
                f.write(tns[i][0]+";"+tns[i][1]+";"+tns[i][2]+";"+tns[i][3]+";"+
                        tns[i][4]+";"+tns[i][5]+";"+tns[i][6]+";"+tns[i][7]+";")
                f.write(fns[i][0]+";"+fns[i][1]+";"+fns[i][2]+";"+fns[i][3]+";"+
                        fns[i][4]+";"+fns[i][5]+";"+fns[i][6]+";"+fns[i][7]+";")
                f.write(p[i][0]+";"+p[i][1]+";"+p[i][2]+";"+p[i][3]+";")
                f.write(r[i][0]+";"+r[i][1]+";"+r[i][2]+";"+r[i][3]+";")
                f.write(f1[i][0]+";"+f1[i][1]+";"+f1[i][2]+";"+f1[i][3]+";")
                f.write(cg[i][0]+";"+cg[i][1]+";"+cg[i][2]+";"+cg[i][3]+";"+
                        cg[i][4]+";"+cg[i][5]+";"+cg[i][6]+";"+cg[i][7]+";")
                f.write(ig[i][0]+";"+ig[i][1]+";"+ig[i][2]+";"+ig[i][3]+";"+
                        ig[i][4]+";"+ig[i][5]+";"+ig[i][6]+";"+ig[i][7]+";")
                f.write(ng[i][0]+";"+ng[i][1]+";"+ng[i][2]+";"+ng[i][3]+";"+
                        ng[i][4]+";"+ng[i][5]+";"+ng[i][6]+";"+ng[i][7])   
                f.write("\n")
        
if __name__ == '__main__':
    main()