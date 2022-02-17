import numpy as np
import glob
import os

def comp(a):
    return float(os.path.splitext(a)[0].split("/")[-1].split("=")[-1])


def main():
    folder = "ct/csv"
    output_folder = "ct/tables"

    result_dict = {}

    files = sorted(glob.glob(folder+"/*.csv"))

    print(files)

    for file in files:
        f = open(file,"r")

        first_line = np.array(f.readline().strip().split(";"))
        initial_order = first_line[0:5]
        parameter_list = first_line[5:]
        results = []
        
        line = f.readline()
        while line:
            split_line = np.array(line.strip().split(";"),dtype="<U9")
            results.append(split_line)
            line = f.readline()

        order = np.array(results)[:,0:5]
        results = np.array(results).transpose()[5:]
        

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
            
            #order[n].tolist()
            result_file.write("->".join(cur_order)+";")
            result_file.write(";".join(r.tolist())+"\n")
        

            

if __name__ == '__main__':
    main()