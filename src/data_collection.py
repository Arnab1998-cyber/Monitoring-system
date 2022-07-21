import cv2
import dlib
from imutils import face_utils
import numpy as np
import os
from utils.all_utils import log, read_yaml, create_directory
import argparse

class data_collect:
    def __init__(self,config,name):
        self.config_file = config
        self.content = read_yaml(self.config_file)
        data_path = self.content['base']['data_dir']
        self.num_image = self.content['base']['number_of_image']
        self.name = name
        self.data_path = os.path.join(data_path,self.name)
        log_dir = self.content['base']['log_dir']
        log_file = self.content['base']['log_file']
        self.log_file = os.path.join(log_dir, log_file)
        self.detector = dlib.get_frontal_face_detector()

    def detect_face(self,image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output = self.detector(gray,1)
            x,y,w,h = face_utils.rect_to_bb(output[0])
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)
            return image
        except:
            return image

    def cropped_image(self,image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output = self.detector(gray,1)
            x,y,w,h = face_utils.rect_to_bb(output[0])
            image = image[y:y+h, x:x+w]
            return image
        except:
            return image

    def open_webcam(self, video_path=0):
        if not os.path.exists(self.data_path):
            create_directory([self.data_path])
        cap = cv2.VideoCapture(video_path)
        for i in range(self.num_image+1):
            ret, frame = cap.read()
            image = frame.copy()
            
            cv2.putText(frame, 'COLLECTING {}/{} of {}'.format(i+1,self.num_image,self.name), (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.waitKey(1000)

            frame = self.detect_face(frame)
            if i != 0:
                image = self.cropped_image(image)

            filename = os.path.join(self.data_path,self.name) + "_{}.jpg".format(i)
            if i != 0:
                cv2.imwrite(filename, image)

            cv2.imshow('feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        log('data collected for {}'.format(self.name), self.log_file)
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    config_path = os.path.join('src','config','config.yaml')
    content = read_yaml(config_path)
    log_dir = content['base']['log_dir']
    log_file = content['base']['log_file']
    file = os.path.join('src',log_dir, log_file)

    args = argparse.ArgumentParser()
    args.add_argument('--params','--p', default = config_path)
    args.add_argument('--name','--n', default = 'Arnab')
    parsed_args = args.parse_args()

    log('data collection started', file)
    app = data_collect(parsed_args.params, parsed_args.name)
    app.open_webcam()
    log('data collection finished',file)
    log('stage 01 completed', file)


