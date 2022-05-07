import numpy as np
import glob
import os

models = {}
models['hybrid_task_cascade_mask_rcnn_X-101-64x4d-FPN'] = '(a)'
models['detectors_htc_r101_20e_coco'] = '(b)'
models['cascade_mask_rcnn_X-101-64x4d-FPN'] = '(c)'
models['cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco'] = '(d)'
models['gcnet_X-101-FPN_DCN_Cascade_Mask_GC(c3-c5,r4)'] = '(e)'

models['mask_rcnn_r50_fpn_1x_coco.py'] = 'mask_rcnn_r50_fpn_1x_coco'
models['mask_rcnn_r50_fpn_1x_cityscapes'] = 'mask_rcnn_r50_fpn_1x_cityscapes'

def comp(a):
    return float(os.path.splitext(a)[0].split("/")[-1].split("=")[-1])


def main():
    folder = "transfer_learning/csv"
    output_folder = "transfer_learning/tables"

    result_dict = {}

    files = sorted(glob.glob(folder+"/*.csv"))

    print(files)

    for file in files:
        f = open(file,"r")

        first_line = np.array(f.readline().strip().split(";"))
        initial_order = first_line[0:2]
        parameter_list = first_line[2:]
        results = []
        
        line = f.readline()
        while line:
            split_line = np.array(line.strip().split(";"),dtype="<U9")
            results.append(split_line)
            line = f.readline()

        order = np.array(results)[:,0:2]
        results = np.array(results).transpose()[2:]
        

        result_dict[os.path.splitext(file)[0].split("/")[-1]] = results

    final = np.dstack(list(result_dict.values()))

    ensembles = "network_order;"+";".join(result_dict.keys())+"\n"

    for f in glob.glob(output_folder+"/*"):
        os.remove(f)

    for nr,param in enumerate(parameter_list): 
        result_file = open(output_folder+"/"+param+".csv","w")
        result_file.write(ensembles)
        for n,r in enumerate(final[nr]):
            cur_order = list(zip(initial_order,order[n]))
            cur_order.sort(key=lambda x:x[1])
            while cur_order[0][1] == "0":
                cur_order.pop(0)
            cur_order = list(zip(*cur_order))[0]
                   
            print(param,cur_order)
            name_switch = []
            for cur in cur_order:
                name_switch.append(models[cur])
            
            #order[n].tolist()
            #result_file.write("->".join(cur_order)+";")
            result_file.write(">".join(name_switch)+";")
            result_file.write(";".join(r.tolist())+"\n")
        

            

if __name__ == '__main__':
    main()