from keras_vggface.utils import preprocess_input
from keras.models import load_model
from src.utils.all_utils import read_yaml, log
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from scipy.spatial import distance
import imutils
import os
import dlib
from imutils import face_utils
import numpy as np
import argparse
import pyttsx3 as p

class prediction:
    def __init__(self, config_path, tracking=None):
        self.tracking = tracking
        config = read_yaml(config_path)
        self.base = config['base']

        artifacts_dir = self.base['artifacts_dir']
        pickle_data_dir = self.base['pickle_format_data_dir']
        image_pkl_file = self.base['img_pickle_file_name']
        raw_data = os.path.join('src',artifacts_dir, pickle_data_dir, image_pkl_file)

        feature_extraction_dir = self.base['feature_extraction_dir']
        feature_files = self.base['extracted_features_name']
        embeddings = os.path.join('src',artifacts_dir, feature_extraction_dir, feature_files)

        model_dir = self.base['model_dir']
        model_file = self.base['model_file']
        self.model_path = os.path.join('src',model_dir,model_file)
        eye_detector_file = self.base['eye_detector_file']
        self.eye_detector_path = os.path.join('src',model_dir,eye_detector_file)

        self.feature_list = pickle.load(open(embeddings,'rb'))
        self.feature_names = pickle.load(open(raw_data,'rb'))

        self.detector = dlib.get_frontal_face_detector()
        self.model = load_model(self.model_path)
        self.eye_detector = dlib.shape_predictor(self.eye_detector_path)
  
        self.thresh = self.base['eye_closing_threshold']
        self.flag = 0
        self.frame_check = self.base['max_closed_eye_frame']

        self.engine = p.init()
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate',180)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice',voices[1].id)
    

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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = self.detector(gray,1)
        x,y,w,h = face_utils.rect_to_bb(output[0])
        image = image[y:y+h, x:x+w]
        return image

    def extract_feature(self,image):
        face = self.cropped_image(image)
        #img = Image.fromarray(face)
        img = cv2.resize(face,(224, 224))
        face_array = np.asarray(img)
        face_array = face_array.astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = self.model.predict(preprocessed_img).flatten()
        return result

    def recommendation(self,feature_list, features):
        similarity = []
        features = np.expand_dims(np.array(features),axis=0)
        #print(len(self.feature_list))
        #print(features,features.shape)
        for i in range(len(feature_list)):
            score = cosine_similarity(features, [feature_list[i]])[0][0]
            similarity.append(score)
        index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
        print(index_pos)
        #print(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1]))
        return index_pos

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def get_avg_aspect_ratio(self,image):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        frame1 = image.copy()
        #frame1 = imutils.resize(frame1, width=450)
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        faces  = self.detector(gray)
        for points in faces:
            shape = self.eye_detector(gray, points)
            shape = face_utils.shape_to_np(shape) #converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
        return ear, leftEyeHull, rightEyeHull

        

    def openWebcam(self, video_path=0):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                image = self.detect_face(frame)
                feature = self.extract_feature(frame)
                index_pos = self.recommendation(self.feature_list,feature)
                name = self.feature_names[index_pos].split('\\')[1].split('_')[0]
                #print(name)
                cv2.putText(image, name, (15,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                if self.tracking:
                    ear, left_eye, right_eye = self.get_avg_aspect_ratio(frame)
                    cv2.drawContours(image, [left_eye], -1, (0, 255, 0), 1)
                    cv2.drawContours(image, [right_eye], -1, (0, 255, 0), 1)
                    if ear < self.thresh:
                        self.flag += 1
                        if self.flag > self.frame_check:
                            cv2.putText(image, "**********************ALERT!**********************", (10, 30),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            self.engine.say("Hello {}! You are sleeping.".format(name))
                            self.engine.runAndWait()
                    else:
                        self.flag = 0

            except Exception as e:
                print(e)
                image = frame
            cv2.imshow('feed',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
    parsed_args = args.parse_args()

    log('Its Monitoring time',file)
    app = prediction(parsed_args.parms)
    app.openWebcam()



