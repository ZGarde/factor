# %load main.py
"""
Created on Thu May  4 21:25:55 2023

@author: 14110
"""

import pandas as pd
from tqdm import tqdm
import 因子复现 as factor
import os


start_date = "20110201"
end_date = "20211201"

#这下面要把file_path改成main文件的位置(主要因为spyder初始地址不一定是文件位置)

file_path = r"C:\Users\ZGarde\Documents\因子复现0518\main.py"
directory = os.path.dirname(file_path)

init_path = r"E:\\main\\feature.txt"
target_path = "./new-feature5.txt"


#阅读feature文件的内容保存至df1中，name代表所有有需要的因子名称
df1 = pd.read_csv(init_path, sep='\t')
df1 = df1.set_index("Unnamed: 0")
name = df1.columns


def deal(df1):
    for stock_name in df1.columns:
        col = df1.loc[:,stock_name]
        start = col.first_valid_index()
        if start == None:
            continue
        end = col.last_valid_index()
        df1.loc[start:end, stock_name] = df1.loc[start:end, stock_name].fillna(method='ffill')
    return df1

#之前读取的文件的数据是str格式，以下代码将代码变成float格式
name = df1.columns.values[2:]
save = {}
for i in tqdm(name):
    small = df1[["Date","Code", i]]
    a2 = list(small.groupby("Code"))
    save[i] = pd.concat(list(map(lambda a: a[1].set_index("Date").drop("Code", axis=1).loc[:end_date], a2)), axis=1)
    save[i].columns = [i[0] for i in list(a2)]
    save[i] = deal(save[i]).astype(float)
save['close'] = save["Close"]

# 根据因子库代码计算因子
for fac in factor.class_name:
    self = fac()
    save[self.name] = self.run(save)
    print(self.name, "finish!")




    
#存储所有需要保存因子的名称
save_name = []
for i in save:
    if i == "close":
        continue
    save_name.append(i)
    
#mark代表行数，以下代码将对应因子写入new-feature.txt文件中,写入逻辑按照股票和日期分别进行遍历
mark = 0
with open(target_path, "w") as f1:
    f1.write("\tDate\tCode")
    for i in save_name:
        if i == "close":
            i = "Close"
        f1.write(f"\t{i}")
    f1.write("\n")
    
    for stock_name in tqdm(save["Close"].columns):
        for date in save["Close"].index:
            if date < int(start_date) or date > int(end_date):
                continue
            save2 = []
            save2.append(str(date))
            save2.append(str(stock_name))
            for factor_name in save_name:
                if save[factor_name].loc[date,stock_name] > 0 or save[factor_name].loc[date,stock_name] <= 0:

                    save2.append("%.4f"%save[factor_name].loc[date,stock_name])
                else:
                    save2.append("")
            f1.write(f"{mark}\t")
            mark += 1
            f1.write("\t".join(save2))
            f1.write("\n")
        f1.flush()  