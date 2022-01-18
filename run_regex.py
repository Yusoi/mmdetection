import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse result files to obtain the highest value of the mask AP")
    parser.add_argument('file', help="File to resolve")
    args = parser.parse_args()

    result_file = open(args.file,"r").read()
    model_names = re.findall(r"Model Order: ([^\n]+)",result_file)
    ap_results = re.findall(r"person[^0-9]*([0-9\.]+)",result_file)

    max_box_ap = 0.0
    max_mask_ap = 0.0
    box_name = ""
    mask_name = ""

    for i in range(0,len(model_names)):
        if float(ap_results[i*2]) > max_box_ap:
            max_box_ap = float(ap_results[i*2])
            box_name = model_names[i]
        if float(ap_results[i*2+1]) > max_mask_ap:
            max_mask_ap = float(ap_results[i*2+1])
            mask_name = model_names[i]
    
    print("Box\n"+box_name+"\n"+str(max_box_ap)+"\n")
    print("Mask\n"+mask_name+"\n"+str(max_mask_ap)+"\n")
        

if __name__ == '__main__':
    main()