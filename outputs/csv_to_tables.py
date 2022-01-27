import numpy as np
import glob
import os

def comp(a):
    return float(os.path.splitext(a)[0].split("/")[-1].split("=")[-1])


def main():
    folder = "aa_vc/csv"
    output_folder = "aa_vc/tables"

    result_dict = {}

    files = sorted(glob.glob(folder+"/*.csv"))

    print(files)

    for file in files:
        f = open(file,"r")

        first_line = np.array(f.readline().strip().split(";"))
        order = first_line[0:10]
        parameter_list = first_line[10:]
        results = []
        
        line = f.readline()
        while line:
            split_line = np.array(line.strip().split(";"),dtype="<U9")
            results.append(split_line)
            line = f.readline()

        order = np.array(results)[:,0:10]
        results = np.array(results).transpose()[10:]
        

        result_dict[os.path.splitext(file)[0].split("/")[-1]] = results

    final = np.dstack(list(result_dict.values()))

    ensembles = ";"+";".join(result_dict.keys())+"\n"

    for f in glob.glob(output_folder+"/*"):
        os.remove(f)

    for nr,param in enumerate(parameter_list): 
        result_file = open(output_folder+"/"+param+".csv","w")
        result_file.write(ensembles)
        for n,r in enumerate(final[nr]):
            result_file.write("-".join(order[n].tolist())+";"+";".join(r.tolist())+"\n")
        

            

if __name__ == '__main__':
    main()