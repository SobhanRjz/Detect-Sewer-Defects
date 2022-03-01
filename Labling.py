import json
import os
import matplotlib.pyplot as plt
import glob
from glob import iglob
import shutil
category=['Deposit', 'OpenJoint' , 'Washing', 'Spalling', 'Deformation', 'AttachedDeposit']
Path = "H:\Video\TrainTest\TrainSewer"
DictData = {}
switcher = {
    '1': 'Deposit',
    '2': 'OpenJoint',
    '3': 'Washing',
    '4': 'Spalling',
    '5': 'Deformation',
    '6': 'AttachedDeposit',
    'n': 'Nothing'
}
filename = "H:\Video\PyProject\Labeled.json"
index = 0
with open(filename, "a+") , open(filename, "r") as file:
    try :
        data = json.load(file)
        index = max([ int(x) for x in data.keys()])
    except:  
        data = {}
initial_count = 0
dir = Path

#Count Image Number
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        initial_count += 1
print(initial_count)

#Should Remove 
# TempList = []
# for i,image in enumerate(glob.glob('H:\Video\TrainTest\TrainSewer\Labeled/*.jpg')):
#     imgnm =str(image).split('\\')[-1]
#     TempList.append(imgnm)
# exist = False;
# for name in TempList:
#     exist = False
#     for dicname in data.values():
#         if dicname["Name"] == name:
#             exist = True
#             break
#     if exist == False:
#         shutil.move(f"{Path}\Labeled\{name}", f"{Path}\Labeled\dLabeled")
imageobj = None
for i,image in enumerate(glob.glob('H:\Video\TrainTest\TrainSewer/*.jpg')):
    index += 1
    photo = plt.imread(image)
    
    if imageobj is None:
        imageobj = plt.imshow(photo)
    else:
        imageobj.set_data(photo)
        
    plt.pause(0.05)
    title_obj = plt.title(str(image).split('\\')[-1]) #get the title property handler
   # plt.getp(title_obj)                    #print out the properties of title
    #plt.pause(0.01)
    inputCat = input('category: Deposit = 1 , OpenJoint = 2 , Washing = 3 , Spalling = 4 , Deformation = 5 , AttachedDeposit = 6 , Nothing = n ==>')
    if inputCat == "end":
        # 1. Read file contents
        data.update(DictData)
        with open(filename, "w") as file:
            json.dump(data, file, indent = 4, sort_keys = True)
        break
    TempList = []
    for item in list(inputCat): TempList.append(item)
    ChosenList = []
    for item in TempList: ChosenList.append(switcher.get(item))
    data[str(index)] = {"Name": str(image).split('\\')[-1], "Categories": ChosenList}
    # del(imageobj)
    # del(photo)
    # del(title_obj)
    shutil.move(f"{image}", f"{Path}\Labeled")
data.update(DictData)
with open(filename, "w") as file:
    json.dump(data, file, indent = 4, sort_keys = True)
