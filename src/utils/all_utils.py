import os
import cv2
import logging as lg
import yaml



def read_yaml(file_path):
    with open(file_path, 'r') as f:
        content=yaml.safe_load(f)
    return content

def create_directory(dirs:list):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

"""def detect_face(image):
    detector = MTCNN()
    output = detector.detect_faces(image)
    x,y,width,height = output[0]['box']
    cv2.rectangle(image, (x,y), (x+width,y+height), color=(0,255,0), thickness=2)
    return image

def cropped_image(image):
    detector = MTCNN()
    output = detector.detect_faces(image)
    x,y,width,height = output[0]['box']
    image = image[y:y+height, x:x+width]
    return image"""

def log(msg,file):
    lg.basicConfig(filename=file, filemode='a', level=lg.INFO,
    format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', datefmt='%d/%m/%Y  %I:%M:%S %p')
    lg.info(msg)

