# Face recognition and Monitoring System

This is a Desktop based application which can recognize you and monitor you, wheather you are sleeping or not.
It is helpful for those students who can not concentrate on their studies and fall asleep during their study.
## Algorithm Used

Keras-Vggface
## Installation

* conda create --prefix ./env python==3.8 -y
* conda activate <<path_to_env_directory>>/env
* pip install -r requirements.txt
## Project Demo

### View my project
[Demo](https://www.linkedin.com/posts/arnab-mitra-882756227_computervision-deeplearning-connections-activity-6955897884059791361-C7Ce?utm_source=linkedin_share&utm_medium=member_desktop_web)

## Steps

### First go to the project directory and run 'python src/app.py' in command line.

### Take image through webcam (by default this application takes 50 images of you)
![Take image](take_image.png)

### Train Model (system collects face embeddings of all persons, present in your data directory)
![Train Model](collection_embeddings.png)

### Prediction or Monitor (In prediction, system only recognize you. In Monitor, it recognize you and monitor you
![demo](demo.png)
