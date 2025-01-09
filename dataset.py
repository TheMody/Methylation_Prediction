

import torch
import numpy as np
from dataset_info import *
from config import *
import xml.etree.ElementTree as ET
import os.path
import pickle 
def preprocess_ds(name, interesting_values ):
    dataset_path = "methylation_data/"+name+"_family.xml"
        #read xml file

    tree = ET.parse(dataset_path + "/"+name+"_family.xml")
    root = tree.getroot()

    samples = []
    interesting_values =interesting_values#["disease"]#["gender", "age", "diseasestatus"]"gender","gender", "age", 

    for child in root:
        sample = {}
        if "Sample" in child.tag:
            sample["id"] = child.get("iid")
            for sample2 in child:
                if "Channel" in sample2.tag:
                    for char in sample2:
                        if "Characteristics" in char.tag:
                            for value in interesting_values:
                                if value in char.attrib["tag"]:
                                    sample[value] = char.text.strip()
            if len(sample) == len(interesting_values) + 1:
                samples.append(sample)
    #check if all ids really exists
    samples = [sample for sample in samples if os.path.isfile(dataset_path + "/" + sample["id"] + "-tbl-1.txt")]
    #check for datapoints which are for some reason not the expected length of 27578
    samples =[sample for sample in samples if (len(open(dataset_path + "/" + sample["id"] + "-tbl-1.txt").readlines())== num_inputs_original)]
    print("length of dataset after filtering", len(samples))
    strtoindex = {}
    for value in interesting_values:
        strtoindex[value] = np.unique([sample[value] for sample in samples]).tolist()
        print(strtoindex[value])

    for sample in samples:
        for value in interesting_values:
            sample[value] = strtoindex[value].index(sample[value])

    #save samples using pickle:
    with open("methylation_data/"+name+"_family.pkl", "wb") as f:
        pickle.dump(samples, f)


class Methylation_ds(torch.utils.data.Dataset):
    def __init__(self, name = "GSE41037", interesting_values = ["disease"]):
      #  self.dataset_path = 'methylation_data/GSE41037_family.xml'
       # self.dataset_path = 'methylation_data/GSE41169_family.xml'
        self.dataset_path = "methylation_data/"+name+"_family.xml"
        self.interesting_values =interesting_values
        #read xml file
        if not os.path.isfile("methylation_data/"+name+"_family.pkl"):
            preprocess_ds(name, interesting_values = interesting_values)
        with open("methylation_data/"+name+"_family.pkl", "rb") as f:
            self.samples = pickle.load(f)

        print("length of dataset", len(self.samples))
        

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        id = self.samples[idx]["id"]

        path_to_sample = self.dataset_path +"/"+ id + "-tbl-1.txt"

       # print(path_to_sample)
        with open(path_to_sample) as f:
            lines = f.readlines()
            x = []
            for line in lines:
                label_value_pair = line.split("\t")
                try:
                    value = float(label_value_pair[1].strip())
                except:
                    value = 0.0
                x.append(value)
            x = np.asarray(x)
        x = torch.tensor(x, dtype=torch.float)
        x = torch.clip(x, 0.0, 1.0)
        x = torch.nn.functional.pad(x, (0, pad_size - len(x)))
        #print(x.shape)
        if len(self.interesting_values) == 0:
            return x, torch.tensor(0, dtype=torch.long)
        return x, torch.tensor(self.samples[idx][self.interesting_values[0]], dtype=torch.long) 


if __name__ == "__main__":
    ds = Methylation_ds(name="GPL8490",interesting_values=[])
 #   print(len(ds))

    for x,y in ds:
        print(torch.min(x), torch.max(x))
        print(y)

    split = int(0.9 * len(ds))
    #train_dataset, val_dataset = torch.utils.data.random_split(ds, [split, len(ds) - split], generator=torch.Generator().manual_seed(seed))

    #calculate baseline using linear regression
    from sklearn.linear_model import LinearRegression, LogisticRegression

    x = []
    y = []
    for i in range(len(ds)):
        x.append(ds[i][0].numpy())
        y.append(ds[i][1][0].numpy())

    x = np.stack(x)
    y = np.stack(y)

    from utils import crossvalidate
    reg = LinearRegression()
    print("crossvalidated accuracy linear regression:",crossvalidate(reg, x, y))

    reg = LogisticRegression()
    print("crossvalidated accuracy logistic regression:",crossvalidate(reg, x, y))


    #calculate proportions of each class
    ys = [0,0,0]
    for x,y in ds:
        ys[y[0]] += 1
    print(ys)
    print([y/len(ds) for y in ys])
        
        
