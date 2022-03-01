# Convert Json labele to Csv
from genericpath import exists
import json
import csv
import os


def ReadJsonFile(Path):
    with open(Path, "r") as file:
        try :
            data = json.load(file)
        except:  
            data = {}
    return data

def WriteData2CSV(Path, data):
    with open (Path, 'w+') as f :
        writer = csv.writer(f)
        writer.writerow(GetHeader())
        writer.writerows(data)
    
def PrepareDataMatrix(data):
    MatrixList = []
    Catagories = (GetHeader())[1:]
    ImageProp = [0] * (Catagories.count() + 1)
    for item in data:
        Name = item["Name"]
        ImageProp[0] = Name
        Catindex = [Catagories.index(i) if i in Catagories else None for i in item["Categories"]]
        for i in Catindex: ImageProp[i] = 1
        MatrixList.append(ImageProp) 

def GetHeader():
    return ['Image Name', 'Deposit', 'OpenJoint' , 'Washing', 'Spalling', 'Deformation', 'AttachedDeposit']

def Main():
    JsonPath = "H:\Video\PyProject\Labeled.json"
    CsvFPath = "H:\Video\PyProject\CSVLabeled.csv"
    data = ReadJsonFile(JsonPath)

    WriteData2CSV(CsvFPath, data)


if __name__ == "__main__":
    Main()
