# canny_filter.py

import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

p = transforms.Compose([transforms.Resize((96, 96)),
                        transforms.ToTensor(),
                        ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
speech_detector = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=(3, 3)),
    nn.Conv2d(16, 64, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(64, 128, kernel_size=(3, 3)),
    nn.Conv2d(128, 256, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.LazyLinear(512),
    nn.LazyLinear(1)
).to(device)
speech_detector.load_state_dict(torch.load("model2850.pt"))
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def predict_image(image):
    image_tensor = p(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = speech_detector(input)
    index = output.data.cpu().numpy() >= 0.7
    return index

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # # is this the right formatting
        frame = frame.to_ndarray(format="bgr24")
        # frame = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        print("Frame processed", 1)
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print("Frame processed", 2)
        image = ""
        print(faces)
        if len(faces) == 0:
            print("Frame processed", 4)
            image = frame
        else:
            (x, y, w, h) = faces[0]
            print("Frame processed", 3)
            if predict_image(Image.fromarray(frame[y:y+h,x:x+w])):
                image = cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                image = frame
            print(predict_image(Image.fromarray(frame[y:y+h,x:x+w])))

        # ret, buffer = cv2.imencode('.jpg', image)
        # frame = buffer.tobytes()

        return image


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)