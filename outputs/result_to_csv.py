import re
import argparse
import os

model_dict = {}
model_dict['mask_rcnn_X-101-64x4d-FPN'] = (('configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py',
                                            'checkpoints/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth',
                                            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth'))
model_dict['cascade_mask_rcnn_X-101-64x4d-FPN'] = (('configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py',
                                                    'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth',
                                                    'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth'))
model_dict['hybrid_task_cascade_mask_rcnn_X-101-64x4d-FPN'] = (('configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py',
                                                                'checkpoints/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth',
                                                                'https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth'))
model_dict['gcnet_X-101-FPN_DCN_Cascade_Mask_GC(c3-c5,r4)'] = (('configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco.py',
                                                                'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth',
                                                                'https://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20210615_161851-720338ec.pth'))
model_dict['scnet_X-101-64x4d-FPN'] = (('configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py',
                                        'checkpoints/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth',
                                        'https://download.openmmlab.com/mmdetection/v2.0/scnet/scnet_x101_64x4d_fpn_20e_coco/scnet_x101_64x4d_fpn_20e_coco-fb09dec9.pth'))
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
        ap_results = re.findall(r"person[^0-9]*([0-9\.]+)",result_file)
        tps = re.findall(r"True Positives (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        fps = re.findall(r"False Positives (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        tns = re.findall(r"True Negatives (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        fns = re.findall(r"False Negatives (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        cg = re.findall(r"Correct Guesses (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        ig = re.findall(r"Incorrect Guesses (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)
        ng = re.findall(r"Not Guessed (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+) (.+)",result_file)

        with open("csv_results_coco/"+os.path.splitext(file)[0].split("/")[-1]+".csv","w") as f:
            for model in key_list:
                f.write(model+";")
            f.write("bbox_ap;mask_ap;tp;tp%;tp_s;tp_s%;tp_m;tp_m%;tp_l;tp_l%;tp_2;tp_2%;tp_2_s;tp_2_s%;tp_2_m;tp_2_m%;tp_2_l;tp_2_l%;fp;fp%;fp_s;fp_s%;fp_m;fp_m%;fp_l;fp_l%;fp_2;fp_2%;fp_2_s;fp_2_s%;fp_2_m;fp_2_m%;fp_2_l;fp_2_l%;tn;tn%;tn_s;tn_s%;tn_m;tn_m%;tn_l;tn_l%;tn_2;tn_2%;tn_2_s;tn_2_s%;tn_2_m;tn_2_m%;tn_2_l;tn_2_l%;fn;fn%;fn_s;fn_s%;fn_m;fn_m%;fn_l;fn_l%;fn_2;fn_2%;fn_2_s;fn_2_s%;fn_2_m;fn_2_m%;fn_2_l;fn_2_l%;cg;cg%;cg_s;cg_s%;cg_m;cg_m%;cg_l;cg_l%;c_cg;c_cg%;c_cg_s;c_cg_s%;c_cg_m;c_cg_m%;c_cg_l;c_cg_l%;ig;ig%;ig_s;ig_s%;ig_m;ig_m%;ig_l;ig_l%;c_ig;c_ig%;c_ig_s;c_ig_s%;c_ig_m;c_ig_m%;c_ig_l;c_ig_l%;ng;ng%;ng_s;ng_s%;ng_m;ng_m%;ng_l;ng_l%;c_ng;c_ng%;c_ng_s;c_ng_s%;c_ng_m;c_ng_m%;c_ng_l;c_ng_l%\n")

            ap_ident = len(model_names) - int(len(ap_results)/2)
            bc_ident = len(model_names) - len(tps)
            print("AP_IDENT:",ap_ident)
            print("BC_IDENT:",bc_ident)

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

                if i < ap_ident:
                    f.write("0.0;0.0;")
                else:
                    f.write(ap_results[2*(i-ap_ident)]+";"+ap_results[2*(i-ap_ident)+1]+";")

                if i < bc_ident:
                    for j in range(0,111):
                        f.write("0.0;")
                    f.write("0.0")
                else:
                    f.write(tps[i-bc_ident][0]+";"+tps[i-bc_ident][1]+";"+tps[i-bc_ident][2]+";"+tps[i-bc_ident][3]+";"+
                            tps[i-bc_ident][4]+";"+tps[i-bc_ident][5]+";"+tps[i-bc_ident][6]+";"+tps[i-bc_ident][7]+";"+
                            tps[i-bc_ident][8]+";"+tps[i-bc_ident][9]+";"+tps[i-bc_ident][10]+";"+tps[i-bc_ident][11]+";"+
                            tps[i-bc_ident][12]+";"+tps[i-bc_ident][13]+";"+tps[i-bc_ident][14]+";"+tps[i-bc_ident][15]+";")
                    f.write(fps[i-bc_ident][0]+";"+fps[i-bc_ident][1]+";"+fps[i-bc_ident][2]+";"+fps[i-bc_ident][3]+";"+
                            fps[i-bc_ident][4]+";"+fps[i-bc_ident][5]+";"+fps[i-bc_ident][6]+";"+fps[i-bc_ident][7]+";"+
                            fps[i-bc_ident][8]+";"+fps[i-bc_ident][9]+";"+fps[i-bc_ident][10]+";"+fps[i-bc_ident][11]+";"+
                            fps[i-bc_ident][12]+";"+fps[i-bc_ident][13]+";"+fps[i-bc_ident][14]+";"+fps[i-bc_ident][15]+";")
                    f.write(tns[i-bc_ident][0]+";"+tns[i-bc_ident][1]+";"+tns[i-bc_ident][2]+";"+tns[i-bc_ident][3]+";"+
                            tns[i-bc_ident][4]+";"+tns[i-bc_ident][5]+";"+tns[i-bc_ident][6]+";"+tns[i-bc_ident][7]+";"+
                            tns[i-bc_ident][8]+";"+tns[i-bc_ident][9]+";"+tns[i-bc_ident][10]+";"+tns[i-bc_ident][11]+";"+
                            tns[i-bc_ident][12]+";"+tns[i-bc_ident][13]+";"+tns[i-bc_ident][14]+";"+tns[i-bc_ident][15]+";")
                    f.write(fns[i-bc_ident][0]+";"+fns[i-bc_ident][1]+";"+fns[i-bc_ident][2]+";"+fns[i-bc_ident][3]+";"+
                            fns[i-bc_ident][4]+";"+fns[i-bc_ident][5]+";"+fns[i-bc_ident][6]+";"+fns[i-bc_ident][7]+";"+
                            fns[i-bc_ident][8]+";"+fns[i-bc_ident][9]+";"+fns[i-bc_ident][10]+";"+fns[i-bc_ident][11]+";"+
                            fns[i-bc_ident][12]+";"+fns[i-bc_ident][13]+";"+fns[i-bc_ident][14]+";"+fns[i-bc_ident][15]+";")
                    f.write(cg[i-bc_ident][0]+";"+cg[i-bc_ident][1]+";"+cg[i-bc_ident][2]+";"+cg[i-bc_ident][3]+";"+
                            cg[i-bc_ident][4]+";"+cg[i-bc_ident][5]+";"+cg[i-bc_ident][6]+";"+cg[i-bc_ident][7]+";"+
                            cg[i-bc_ident][8]+";"+cg[i-bc_ident][9]+";"+cg[i-bc_ident][10]+";"+cg[i-bc_ident][11]+";"+
                            cg[i-bc_ident][12]+";"+cg[i-bc_ident][13]+";"+cg[i-bc_ident][14]+";"+cg[i-bc_ident][15]+";")
                    f.write(ig[i-bc_ident][0]+";"+ig[i-bc_ident][1]+";"+ig[i-bc_ident][2]+";"+ig[i-bc_ident][3]+";"+
                            ig[i-bc_ident][4]+";"+ig[i-bc_ident][5]+";"+ig[i-bc_ident][6]+";"+ig[i-bc_ident][7]+";"+
                            ig[i-bc_ident][8]+";"+ig[i-bc_ident][9]+";"+ig[i-bc_ident][10]+";"+ig[i-bc_ident][11]+";"+
                            ig[i-bc_ident][12]+";"+ig[i-bc_ident][13]+";"+ig[i-bc_ident][14]+";"+ig[i-bc_ident][15]+";")
                    f.write(ng[i-bc_ident][0]+";"+ng[i-bc_ident][1]+";"+ng[i-bc_ident][2]+";"+ng[i-bc_ident][3]+";"+
                            ng[i-bc_ident][4]+";"+ng[i-bc_ident][5]+";"+ng[i-bc_ident][6]+";"+ng[i-bc_ident][7]+";"+
                            ng[i-bc_ident][8]+";"+ng[i-bc_ident][9]+";"+ng[i-bc_ident][10]+";"+ng[i-bc_ident][11]+";"+
                            ng[i-bc_ident][12]+";"+ng[i-bc_ident][13]+";"+ng[i-bc_ident][14]+";"+ng[i-bc_ident][15])   

                f.write("\n")
        
if __name__ == '__main__':
    main()