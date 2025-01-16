

import torch
import numpy as np
from config import *
import xml.etree.ElementTree as ET
import os.path
import pickle 
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_ds(name, interesting_values ):
    dataset_path = "methylation_data/"+name+"_family.xml"
        #read xml file

    tree = ET.parse(dataset_path + "/"+name+"_family.xml")
    root = tree.getroot()

    samples = []
    interesting_values =interesting_values#["disease"]

    for child in root:
        sample = {}
        if "Sample" in child.tag:
            sample["id"] = child.get("iid")
            for sample2 in child:
                if "Channel" in sample2.tag:
                    for char in sample2:
                        if "Characteristics" in char.tag:
                            for value in interesting_values:
                              #  print(char.attrib["tag"])
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
        print(len(strtoindex[value]))

    for sample in samples:
        for value in interesting_values:
            sample[value] = strtoindex[value].index(sample[value])

    #save samples using pickle:
    with open("methylation_data/"+name+interesting_values[0]+"_family.pkl", "wb") as f:
        pickle.dump(samples, f)


class Methylation_ds(torch.utils.data.Dataset):
    def __init__(self, name = "GSE41037", interesting_values = ["disease"], load_into_mem = True):
      #  self.dataset_path = 'methylation_data/GSE41037_family.xml'
       # self.dataset_path = 'methylation_data/GSE41169_family.xml'
        self.dataset_path = "methylation_data/"+name+"_family.xml"
        self.interesting_values =interesting_values
        #read xml file
        preprocess_name = "methylation_data/"+name+interesting_values[0]+"_family.pkl"
        if not os.path.isfile(preprocess_name):
            preprocess_ds(name, interesting_values = interesting_values)
        with open(preprocess_name, "rb") as f:
            self.samples = pickle.load(f)

        print("length of dataset", len(self.samples))

        self.ds_in_mem = load_into_mem
        if load_into_mem:
            full_ds_name = "methylation_data/"+name+interesting_values[0]+"_family_preprocessed_ds.pkl"
            if not os.path.isfile(full_ds_name):
                self.mem_ds = []
                print("processing dataset")
                for i in tqdm(range(len(self.samples))):
                    self.mem_ds.append(self.get_item_internally(i))
                with open(full_ds_name, "wb") as f:
                    pickle.dump(self.mem_ds, f)
            else:
                with open(full_ds_name, "rb") as f:
                    self.mem_ds = pickle.load(f)
    def get_item_internally(self,idx):
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
        if len(self.interesting_values) == 0:
            return x, torch.tensor(0, dtype=torch.long)
        return x, torch.tensor(self.samples[idx][self.interesting_values[0]], dtype=torch.long) 

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.ds_in_mem:
            return self.mem_ds[idx]
        else:
            return self.get_item_internally(idx)


if __name__ == "__main__":
    ds = Methylation_ds(name = "GPL8490", interesting_values=["disease"] ,load_into_mem=True)

    #calculate proportions of each class
    ys = np.zeros(num_classes)
    for x,y in ds:
        ys[y] += 1

    plt.bar(range(num_classes), ys)
    plt.ylabel("number of samples")
    plt.xlabel("class")
    plt.show()
    print(ys)
    print(ys/len(ds))
        
        
