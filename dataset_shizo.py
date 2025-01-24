

import torch
import numpy as np
from config import *
from lxml import etree
import os.path
import pickle 
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import os
import random

def count_lines_wc(filename):
    """Counts lines using the wc -l command."""
    output = subprocess.check_output(['wc', '-l', filename])
    # wc -l output looks like: b'   50000 filename\n'
    return int(output.strip().split()[0])

def filter_samples_by_line_count_wc(samples, dataset_path, num_inputs_original):
    filtered = []
    for sample in tqdm(samples):
        file_path = os.path.join(dataset_path, sample["id"] + "-tbl-1.txt")
        line_count = count_lines_wc(file_path)
        if line_count == num_inputs_original:
            filtered.append(sample)
    return filtered

control_names = [ '1; control','Control','Nonobese', 'Normal', 'Normal Bone Marrow (control)', 'Normal Lung Tissue sample','Normal adjacent to colorectal adenocarcinoma','Unaffected', 'Unaffected control','adult normal liver', 'control','control (UPPP)', 'control-HPVneg', 'control-HPVpos', 'healthy','healthy control','leukaemia control','non-diabetic', 'non-endometriosis', 'non-small cell lung cancer (NSCLC)', 'none', 'normal','normal adrenal tissue', 'normal brca1 mutation 185 del g', 'normal control', 'normal monozygotic (MZ)', 'normal mucosa','normal/healthy',]

def preprocess_ds(name, interesting_values, clean_names = False ):
    dataset_path = path_to_data+name+"_family.xml"
        #read xml file
    parser = etree.XMLParser(recover=True)  # tries to recover from minor errors
    tree = etree.parse(dataset_path + "/"+name+"_family.xml", parser=parser)
    # parser = ET.XMLParser(encoding="UTF-8")
    # tree = ET.parse(dataset_path + "/"+name+"_family.xml", parser = parser)
    root = tree.getroot()

    samples = []
   # interesting_values =interesting_values#["disease"]

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
                                if "tag" in char.attrib:
                                    if value in char.attrib["tag"]:
                                        sample[value] = char.text.strip()
            if len(sample) == len(interesting_values) + 1:
                samples.append(sample)
    #check if all ids really exists
    print("length of dataset before filtering", len(samples))
    samples = [sample for sample in samples if os.path.isfile(dataset_path + "/" + sample["id"] + "-tbl-1.txt")]
    print("length of dataset after filtering out samples which do not exist", len(samples))
    #check for datapoints which are for some reason not the expected length of 27578
   # print(len(open(dataset_path + "/" + samples[0]["id"] + "-tbl-1.txt").readlines()))
    samples = filter_samples_by_line_count_wc(samples, dataset_path, num_inputs_original)
    #samples =[sample for sample in samples if (len(open(dataset_path + "/" + sample["id"] + "-tbl-1.txt").readlines())== num_inputs_original)]
    print("length of dataset after filtering out samples with inconsistent lengths", len(samples))

    strtoindex = {}
    for value in interesting_values:
        strtoindex[value] = np.unique([sample[value] for sample in samples]).tolist()
        print(strtoindex[value])
        print(len(strtoindex[value]))

    for sample in samples:
        for value in interesting_values:
            sample[value] = strtoindex[value].index(sample[value])
    
    # if clean_names:
    #     duplicate_ids = []
    #     for value in interesting_values:
    #         for name in control_names:
    #             duplicate_ids.append(strtoindex[value].index(name))

    #     for sample in samples:
    #         for value in interesting_values:
    #             if sample[value] in duplicate_ids:
    #                 sample[value] = duplicate_ids[0]

    #save samples using pickle:
    with open("methylation_data/"+name+"_family.pkl", "wb") as f:
        pickle.dump(samples, f)

    return strtoindex

class Methylation_ds(torch.utils.data.Dataset):
    def __init__(self, name = "GSE41037", interesting_values = ["disease"], load_into_mem = True, normalize_ds = True):
      #  self.dataset_path = 'methylation_data/GSE41037_family.xml'
       # self.dataset_path = 'methylation_data/GSE41169_family.xml'
        self.dataset_path = path_to_data +name+"_family.xml"
        self.interesting_values =interesting_values
        #read xml file
        preprocess_name = "methylation_data/"+name+"_family.pkl"
        if not os.path.isfile(preprocess_name):
            preprocess_ds(name, interesting_values = interesting_values)
        with open(preprocess_name, "rb") as f:
            self.samples = pickle.load(f)
        self.normalize_ds = normalize_ds
        print("length of dataset", len(self.samples))

        if normalize_ds:
            if os.path.isfile("methylation_data/"+name+"_family_normalization.pkl"):
                with open("methylation_data/"+name+"_family_normalization.pkl", "rb") as f:
                    running_mean, running_var = pickle.load(f)
            else:
                print("processing dataset for normalization")
                running_mean = torch.zeros(pad_size)
                running_var = torch.zeros(pad_size)
                self.normalize_ds = False
                length = min(1000,len(self.samples)+1)
                for i in tqdm(range(1,length)):
                    #filter for outliers:
                    sample = self.get_item_internally(random.randint(0,len(self.samples)-1))
                    if torch.max(torch.abs(sample[0])) > 1e6  :
                        print("outlier")
                        continue
                    running_mean += sample[0]
                running_mean /= length
                for i in tqdm(range(1,length)):
                    sample = self.get_item_internally(random.randint(0,len(self.samples)-1))
                    if torch.max(torch.abs(sample[0])) > 1e6  :
                        print("outlier")
                        continue
                    running_var += (sample[0]-running_mean)**2
                running_var /= length
                self.normalize_ds = True
                with open("methylation_data/"+name+"_family_normalization.pkl", "wb") as f:
                    pickle.dump((running_mean, running_var), f)
            self.running_mean = running_mean    
            self.running_var = running_var
        self.running_var = torch.sqrt(self.running_var)
        self.running_var[self.running_var == 0] = 1.0
        #check for nan
        self.running_var[torch.isnan(self.running_var)] = 1.0
        # print("running mean", self.running_mean)
        # print("running var", self.running_var)

        self.ds_in_mem = load_into_mem
        if load_into_mem:
            full_ds_name = "methylation_data/"+name+"_family_preprocessed_ds.pkl"
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
        x = torch.nn.functional.pad(x, (0, pad_size - len(x)))
        x[torch.isnan(x)] = 0.0
        if self.normalize_ds:   
            x = (x - self.running_mean)/self.running_var
            #x = torch.clip(x, 0.0, None)
            #filter for weird outliers
            x[x > 20] = 0.0
            x[x < -20] = 0.0
      #     x = torch.clip(x, -10.0, 10.0)
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
    #ds = Methylation_ds(name = "GPL8490", interesting_values=["disease"] ,load_into_mem=True)
    ds = Methylation_ds(name = "GPL570", interesting_values=[],load_into_mem=False)
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
        
        
