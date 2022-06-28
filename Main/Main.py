from importlib.resources import path
from threading import Thread
import sys
import cv2
from queue import Queue
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
# class FileVideoStream:
#     def __init__(self, path, queueSize=128):
#         # initialize the file video stream along with the boolean
#         # used to indicate if the thread should be stopped or not
#         self.stream = cv2.VideoCapture(path)
#         self.stopped = False
#         # initialize the queue used to store frames read from
#         # the video file
#         self.Q = Queue(maxsize=queueSize)
#     def start(self):
#         # start a thread to read frames from the file video stream
#         t = Thread(target=self.update, args=())
#         t.daemon = True
#         t.start()
#         return self
#     def update(self):
#         # keep looping infinitely
#         while True:
#             # if the thread indicator variable is set, stop the
#             # thread
#             if self.stopped:
#                 return
#             # otherwise, ensure the queue has room in it
#             if not self.Q.full():
#                 # read the next frame from the file
#                 (grabbed, frame) = self.stream.read()
#                 # if the `grabbed` boolean is `False`, then we have
#                 # reached the end of the video file
#                 if not grabbed:
#                     self.stop()
#                     return
#                 # add the frame to the queue
#                 self.Q.put(frame)
#     def read(self):
#         # return next frame in the queue
#         return self.Q.get()
#     def more(self):
#         # return True if there are still frames in the queue
#         return self.Q.qsize() > 0
#     def stop(self):
#         # indicate that the thread should be stopped
#         self.stopped = True

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
        frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])
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