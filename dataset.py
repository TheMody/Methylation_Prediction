

import torch
import numpy as np
from datasets import load_from_disk
from dataset_info import *
from config import *
import xml.etree.ElementTree as ET



class Methylation_ds(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset_path = 'methylation_data/GSE41037_family.xml'
       # self.dataset_path = 'methylation_data/GSE41169_family.xml'
       # self.dataset_path = "methylation_data/GPL8490_family.xml"
        #read xml file

        tree = ET.parse(self.dataset_path + "/GSE41037_family.xml")
        root = tree.getroot()

        self.samples = []
        self.interesting_values =["disease"]#["gender", "age", "diseasestatus"]"gender","gender", "age", 

        for child in root:
            sample = {}
            if "Sample" in child.tag:
                sample["id"] = child.get("iid")
                for sample2 in child:
                    if "Channel" in sample2.tag:
                        for char in sample2:
                            if "Characteristics" in char.tag:
                                for value in self.interesting_values:
                                    if value in char.attrib["tag"]:
                                        sample[value] = char.text.strip()
                if len(sample) == len(self.interesting_values) + 1:
                    self.samples.append(sample)
    
        self.strtoindex = {}
        for value in self.interesting_values:
            self.strtoindex[value] = np.unique([sample[value] for sample in self.samples]).tolist()
            print(self.strtoindex[value])
        

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
                #label = label_value_pair[0]
                try:
                    value = float(label_value_pair[1].strip())
                except:
                    value = 0.0
                x.append(value)
            x = np.asarray(x)
        x = torch.tensor(x, dtype=torch.float)
        return x, [torch.tensor(self.strtoindex[v].index(self.samples[idx][v]), dtype=torch.long) for v in self.interesting_values]


if __name__ == "__main__":
    ds = Methylation_ds()
 #   print(len(ds))
    ys = [0,0,0]
    for x,y in ds:
        ys[y[0]] += 1
    print(ys)
    print([y/len(ds) for y in ys])
        
        
