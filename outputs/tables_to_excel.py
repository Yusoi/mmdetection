import numpy as np
import glob
import os
import pandas as pd



def main():
    folder = "cur_tables"

    result_dict = {}

    writer = pd.ExcelWriter("compare_a.xlsx", engine='xlsxwriter')

    _list = glob.glob(folder+"/*.csv")
    _list.sort()

    for file in _list:
        name = os.path.splitext(file)[0].split("/")[-1]
        f = pd.read_csv(file, sep=";")
        f.to_excel(writer, sheet_name=name)
        
    writer.save()

if __name__ == '__main__':
    main()