#IMPORTANT IMPORTS
import torch
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import os
from datetime import datetime
import cv2
import mysql.connector
import pytorchvideo.data
import os
from transformers import VivitImageProcessor, VivitForVideoClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from pytorchvideo.transforms import (
ApplyTransformToKey,
Normalize,
RandomShortSideScale,
RemoveKey,
ShortSideScale,
UniformTemporalSubsample,)
from torchvision.transforms import (
Compose,
Lambda,
RandomCrop,
RandomHorizontalFlip,
Resize,
)
from transformers import AutoTokenizer, AutoModelForVideoClassification
from transformers import pipeline

#Defining Model
model = AutoModelForVideoClassification.from_pretrained("prathameshdalal/vivit-b-16x2-kinetics400-UCF-Crime")
pipe = pipeline("video-classification", model="prathameshdalal/vivit-b-16x2-kinetics400-UCF-Crime")
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
#Necessary Pre-processing Methods
mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)
num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps

#transforms on dataset
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join("T2"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

def run_inference(model, video):
    import torch.nn.functional as F
    # (num_frames, num_channels, height, width)
    permuted_sample_test_video = video.permute(1, 0, 2, 3)
    inputs = {
        "pixel_values": permuted_sample_test_video.unsqueeze(0)  # this can be skipped if you don't have labels available.
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to get class probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Get the class with the highest probability and its confidence score
        max_prob, predicted_class = torch.max(probabilities, 1)

    return max_prob.item(), predicted_class.item()


import imageio
import numpy as np
from IPython.display import Image

def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)

def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.
    
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 1}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename
def display_gif(video_tensor, gif_name="sample.gif"):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)


    # Release the video capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------------------------------------------------------------------------
#_------------------------------------------DO NOT CHANGE ANYTHING ABOVE THIS LINE -----------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------
output_folder = "T2\Fighting"    
def capture_webcam_snippets():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening webcam.")
        return

    fps = 30
    snippet_duration = 2  # in seconds
    snippet_frame_count = snippet_duration * fps

    try:
        while True:
            snippet_frames = []
            for _ in range(snippet_frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                snippet_frames.append(frame)

            if not snippet_frames:
                break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snippet_path = os.path.join(output_folder, f'snippet_{timestamp}.mp4')

            snippet = cv2.VideoWriter(snippet_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
            for snippet_frame in snippet_frames:
                snippet.write(snippet_frame)
            snippet.release()
            
            sample_test_video = next(iter(test_dataset))
            logits = run_inference(model, sample_test_video["video"])
            predicted_class_idx = logits[1]
            risk_level = logits[0]

            nature = model.config.id2label[predicted_class_idx]
            print("Predicted class:", nature, "Risk Level: ", risk_level)
            if nature != 'Normal':
                display_gif(sample_test_video["video"], "temp.gif")
                path = os.path.join("T2", "Fighting", sample_test_video['video_name'])
                print(path)
                vs = cv2.VideoCapture(path)

                # loop over the frames from the video stream
                while True:
                    n = nature
                # grab the frame from the threaded video stream and resize it
                    c = 1
                    ret, frame = vs.read()
                    if not ret or frame is None:
                        print("Error opening video stream or file")
                        break  # Break out of the loop if there's an error or no frame

                    frame = imutils.resize(frame, width=800)

                    # grab the frame dimensions and convert it to a blob
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                        0.007843, (300, 300), 127.5)

                    # pass the blob through the network and obtain the detections and
                    # predictions
                    net.setInput(blob)
                    detections = net.forward()

                    # loop over the detections
                    for i in np.arange(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with
                        # the prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by ensuring the `confidence` is
                        # greater than the minimum confidence
                        if confidence > 0.4:
                            # extract the index of the class label from the
                            # `detections`, then compute the (x, y)-coordinates of
                            # the bounding box for the object
                            idx = int(detections[0, 0, i, 1])
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # draw the prediction on the frame
                            label = "{}: {:.2f}%".format(CLASSES[idx],
                                confidence * 100)
                            if label[0:6]=='person':
                                c += 1
                            if label[0:6]=='person' or label[0:3]=='dog' or label[0:3]=='cat':
                                cv2.rectangle(frame, (startX, startY), (endX, endY),
                                    COLORS[idx], 2)
                                y = startY - 15 if startY - 15 > 15 else startY + 15
                                cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    
                    cv2.putText(frame, f'Total People: {c-1}',(20,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)
                            
                    print(f"Humans in frame: {c-1}")
                    # show the output frame
                    cv2.imshow("Video Feed", frame)
                    key = cv2.waitKey(1) & 0xFF

                    # if the `q` key was pressed, break from the loop
                    if key == ord("p"):
                        sample_test_video = next(iter(test_dataset))
                        break

                # do a bit of cleanup
                cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("Webcam Snippet Capture Stopped.")
    finally:
        cap.release()

            
if __name__ == "__main__":
    capture_webcam_snippets()
