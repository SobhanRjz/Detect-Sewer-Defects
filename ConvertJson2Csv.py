# Convert Json labele to Csv
from genericpath import exists
import json
import csv
from operator import contains
import os

from numpy import matrix


def ReadJsonFile(Path):
    with open(Path, "r") as file:
        try :
            data = json.load(file)
        except:  
            data = {}
    return data

def WriteData2CSV(Path, data):
    with open (Path, 'w+', newline = '') as f :
        writer = csv.writer(f)
        writer.writerow(GetHeader())
        writer.writerows(data)
    
def PrepareDataMatrix(data):
    MatrixList = []
    Catagories = (GetHeader())[1:]
    for item in data:
        ImageProp = [0] * (len(Catagories) + 1)
        Name = data[item]["Name"]
        ImageProp[0] = Name
        Catindex = [Catagories.index(i) if i in Catagories else None for i in data[item]["Categories"]]
        if None in Catindex: continue
        for i in Catindex: ImageProp[i + 1] = 1
        MatrixList.append(ImageProp) 
    return MatrixList

def GetHeader():
    return ['Image Name', 'Deposit', 'OpenJoint' , 'Washing', 'Spalling', 'Deformation', 'AttachedDeposit', 'Nothing']

def Main():
    JsonPath = "H:\Video\PyProject\OutPut\Labeled.json"
    CsvFPath = "H:\Video\PyProject\OutPut\CSVLabeled.csv"
    data = ReadJsonFile(JsonPath)
    MatrixData = PrepareDataMatrix(data)
    WriteData2CSV(CsvFPath, MatrixData)


if __name__ == "__main__":
    Main()
