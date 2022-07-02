import json
from importlib.resources import path
import pickle
from threading import Thread
import sys
import pandas as pd
import cv2
from queue import Queue
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
sys.path.insert(0, 'H:\Video\PyProject\CombineFeatures')
import MainCombineFeatures



def FeatureExtraction(Image :np.ndarray, OutPutName: str, FeatureList: list):
    OutPutPath = r"H:\Video\PyProject\CombineFeatures\OutPuts"
    if _IsDataFrameExist is not True:
        image_dataset = pd.DataFrame()
        i = -1
        FeatureSwitcher = MainCombineFeatures.PythonSwitch()
        # Defect
        
        df = pd.DataFrame()
        for ft in FeatureList:
            DataFeat = FeatureSwitcher.Method(ft, Image)
            df1 = pd.DataFrame(columns = [ft + str(j) for j in range(DataFeat.size)]) 
            df1.loc[0] = DataFeat
            df = pd.concat([df, df1], axis = 1)

        image_dataset = pd.concat([image_dataset, df])
            # y = x.index == 0
            # np.where(np.array(y) == True)[0]
            # if i == 30:
            #     break
        # image_dataset.to_pickle(OutPutPath + "\\" + OutPutName + ".pkl")  

    else :
        image_dataset = pd.read_pickle(OutPutPath + "\\" + OutPutName +".pkl")
        image_dataset = pd.DataFrame(image_dataset).fillna(0)

    return image_dataset

# Extract Best Filter 
JsonPath = 'H:\Video\PyProject\CombineFeatures\OutPuts\BestModelByMaxACC.json'
with open(JsonPath, 'r') as f:
    data = json.load(f)
FiltersName = list(data)[0]
FeatureNameList = FiltersName.split()
MachineLModel = pickle.load(open("H:\Video\PyProject\CombineFeatures\OutPuts\BestModel\RandomForest.sav", 'rb'))
AIModel = keras.models.load_model('H:\Video\PyProject\Model.h5')
LABLES = ["Deposit", "OpenJoint", "Washing", "Spalling", "Deformation", "AttachedDeposit", "Nothing"]
# FeatureList = ["HOG", "SIFT"]

Path = r"H:\Video\PyProject\Main\TestFilm\Test.mp4"
fvs = FileVideoStream(Path, transform = None, queue_size = 128).start()
time.sleep(1.0)
fps = FPS().start()
i=0 #frame counter
frameTime = 1
# loop over frames from the video file stream
while fvs.more():
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    i= i+1 #increment counter
    frame = fvs.read()
    if i % 5 == 0:
        Frame = imutils.resize(frame, width=450)
        Frame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        Frame = np.dstack([Frame, Frame, Frame])
        _IsDataFrameExist = False
        FrameFeature = FeatureExtraction(frame, "Train_" + "_".join(FeatureNameList), FeatureNameList) 
        # SSTrain = StandardScaler()
        # SSTrain.fit(FrameFeature)
        # FrameFeature = SSTrain.transform(FrameFeature)
        Predict = MachineLModel.predict(FrameFeature)
        if (Predict[0] == 0): # Defect
            PrepareFrame =  cv2.resize(Frame, (64, 64))
            PrepareFrame = PrepareFrame.reshape(-1, 64, 64, 3)
            AiPredict = AIModel.predict([PrepareFrame])
            AiPredict[AiPredict>0.5]=1
            AiPredict[AiPredict<=0.5]=0

            print()#Ai
        else:
            print()
            #No Defect

        # display the size of the queue on the frame
        cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
        # show the frame and update the FPS counter
        cv2.imshow("Frame", frame)
        if cv2.waitKey(frameTime) & 0xFF == ord('q'):
            break
        fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()