import pickle
import cv2
import numpy as np
from skimage.io import imread
from skimage.feature import hog
from sklearn.svm import LinearSVC
from PIL import Image, ImageDraw
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3
ModelPath = r'H:\Video\PyProject\HogDetect\ModelOutPut\SvmModel.sav'
model = pickle.load(open(ModelPath, 'rb'))


def DrawRecDefectOrNotDefect(frame, Predict):
    pt1 = (round(frame.shape[1] * 0.3), round(frame.shape[0] * 0.9))
    pt2 = (round(frame.shape[1] * 0.7), round(frame.shape[0] * 0.95))
    cv2.rectangle(frame, pt1 = pt1, pt2 = pt2 , color = (0, 0, 0), thickness=-1)
    if Predict == 0:
        label = "Have not any defect"
    else:
         label = "Have defect"
    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)[0]

    c1 = (int((pt1[0] + pt2[0]) / 2 - (textsize[0] / 2)), int((pt1[1] + pt2[1]) / 2 + (textsize[1] / 2)))
    cv2.putText(frame, label, c1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (215, 185, 49), 2)
    # shape = frame.shape
    # mask = Image.new("RGB", (400, 400), "black")
    # draw = ImageDraw.Draw(mask)

    # # Draw a rounded rectangle
    # draw.rounded_rectangle((0, 50, 100, 0), fill="black", radius= 5)
    # points = np.array([[160, 130], [350, 130], [250, 300]])
    # #cv2.fillPoly(Image, pts=[points], color=(255, 0, 0))
    # s_img = mask
    # l_img = frame
    # x_offset=y_offset=10
    # l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    # cv2.imshow("output", l_img)
    # cv2.waitKey(0)
    # mask.show()
def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

def PredictSVM_Model(Image):        
    fd = hog(Image, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True, channel_axis=2)# fd= feature descriptor
    if fd.shape[0] < model.n_features_in_:
        fd = padarray(fd, model.n_features_in_)
    fd = fd.reshape(1,-1)
    prediction = model.predict(fd)
    return prediction[0]

def main():
    # reading the input
    VideoPath = r"H:\Video\PyProject\HogDetect\VideoTest\5-Test.avi"
    cap = cv2.VideoCapture(VideoPath)
    
    output = cv2.VideoWriter(
        r"H:\Video\PyProject\HogDetect\VideoTest\TestDetectOutPut.avi", cv2.VideoWriter_fourcc(*'MPEG'), 
      30, (1080, 1920))
  
    while(True):
        ret, frame = cap.read()
        if(ret):
            Predict = PredictSVM_Model(frame)
            DrawRecDefectOrNotDefect(frame, Predict)
            # adding filled rectangle on each frame
            # cv2.rectangle(frame, (100, 150), (500, 600),
            #               (0, 255, 0), -1)
              
            # writing the new frame in output
            output.write(frame)
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
  
    cv2.destroyAllWindows()
    output.release()
    cap.release()
  
  
if __name__ == "__main__":
    main()