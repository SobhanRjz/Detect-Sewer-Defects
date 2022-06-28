import pathlib
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Layer
from tensorflow import keras
import tensorflow_addons as tfa
tf.executing_eagerly()

#Read Data
PathCsv = "H:\Video\PyProject\OutPut\CSVLabeled.csv"
df=pd.read_csv(PathCsv)
print(df.head())
LABELS = df.head(0).columns[1:]
LABELS0 = []
for lb in LABELS:
  LABELS0.append(lb)

fnames = df.iloc[:,0]

temp_path = str("H:\Video\TrainTest\TrainSewer\Labeled\\")
pathlist = [temp_path + fr for fr in fnames]
# data_dir = pathlib.Path("H:\Video\TrainTest\TrainSewer\\")
# filename = list(data_dir.glob('Labeled/*.jpg'))
# fnames = []
# for fname in filename:
#   fnames.append(fname)

ds_size= len(fnames)
print("Number of images in folders: ", ds_size)

number_of_selected_samples = ds_size
filelist_ds = tf.data.Dataset.from_tensor_slices(pathlist[:number_of_selected_samples])

ds_size= filelist_ds.cardinality().numpy()
print("Number of selected samples for dataset: ", ds_size)


def get_label(file_path): 
  parts = file_path.split('\\')
  file_name= parts[-1]
  labels= df[df["Image Name"]==file_name][LABELS0].to_numpy().squeeze()
  return tf.convert_to_tensor(labels)

IMG_WIDTH, IMG_HEIGHT = 64 , 64
def decode_img(img):
  #color images
  img = tf.image.decode_jpeg(img, channels=3) 
  #convert unit8 tensor to floats in the [0,1]range
  img = tf.image.convert_image_dtype(img, tf.float32) 
  #resize 
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) 

def combine_images_labels(file_path: tf.Tensor):
  print("file_path: ",bytes.decode(file_path.numpy()),type(bytes.decode(file_path.numpy())))
  label = get_label(bytes.decode(file_path.numpy()))
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

train_ratio = 0.80
ds_train=filelist_ds.take(ds_size*train_ratio)
ds_test=filelist_ds.skip(ds_size*train_ratio)
BATCH_SIZE = 64

# for i in ds_train:
#   combine_images_labels(i)
ds_train=ds_train.map(lambda x: tf.py_function(func=combine_images_labels,
          inp=[x], Tout=(tf.float32,tf.int64)),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)



# for one_element in ds_train:
#     print(one_element)

ds_test= ds_test.map(lambda x: tf.py_function(func = combine_images_labels,
          inp=[x], Tout = (tf.float32,tf.int64)),
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)

def covert_onehot_string_labels(label_string,label_onehot):
  labels=[]
  for i, label in  enumerate(label_string):
    if np.size(label_onehot) != 0 and label_onehot[i]:
       labels.append(label)
  if len(labels)==0:
    labels.append("NONE")
  return labels

def show_samples(dataset):
  fig=plt.figure(figsize=(16, 16))
  columns = 3
  rows = 3
  print(columns*rows,"samples from the dataset")
  i=1
  for a,b in dataset.take(columns*rows): 
    fig.add_subplot(rows, columns, i)
    plt.imshow(np.squeeze(a))
    plt.title("image shape:"+ str(a.shape)+" ("+str(b.numpy()) +") "+ 
              str(covert_onehot_string_labels(LABELS,b.numpy())))
    print(str(covert_onehot_string_labels(LABELS,b.numpy())))
    i=i+1
  plt.show()
# show_samples(ds_test)

ds_train_batched=ds_train.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE) 
ds_test_batched=ds_test.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)


# ds_test_batched = ds_test_batched.batch(BATCH_SIZE).cache().prefetch(tf.data.experimental.AUTOTUNE)

print("Number of batches in train: ", ds_train_batched.cardinality().numpy())
print("Number of batches in test: ", ds_test_batched.cardinality().numpy())









model = keras.models.load_model('Model.h5')
ds= ds_test_batched
print("Test Accuracy: ", model.evaluate(ds)[1])
ds=ds_test
predictions= model.predict(ds.batch(batch_size=10).take(1))
print("A sample output from the last layer (model) ", predictions[0])
y=[]
print("10 Sample predictions:")
for (pred,(a,b)) in zip(predictions,ds.take(10)):
  
  pred[pred>0.5]=1
  pred[pred<=0.5]=0
  print("predicted: " ,pred, str(covert_onehot_string_labels(LABELS, pred)),  
        "Actual Label: ("+str(covert_onehot_string_labels(LABELS,b.numpy())) +")")
  y.append(b.numpy())