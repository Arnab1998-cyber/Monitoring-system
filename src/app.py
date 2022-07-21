import tkinter as tk
from tkinter import *
import argparse
import tkinter.font as font
import webbrowser
import random
import os

from src.utils.all_utils import read_yaml, log
from src.data_collection import data_collect
from src.create_image_pkl import generate_pkl_file
from src.feature_extractor import extractor, feature_extractor
from src.prediction import prediction

class RegistrationModule:
    def __init__(self):
        self.window = Tk()
        self.window.title("Face Recognition and Tracking")

        self.window.resizable(0, 0)
        window_height = 600
        window_width = 880

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))

        self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        # window.geometry('880x600')
        self.window.configure(background='#ffffff')

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        header = tk.Label(self.window, text="Employee Monitoring Registration", width=65, height=2, fg="white", bg="#363e75",
                          font=('times', 18, 'bold', 'underline'))
        header.place(x=0, y=0)

        empID = tk.Label(self.window, text="EmpID", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empID.place(x=450, y=80)
        self.empIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empIDTxt.place(x=575, y=89)

        empName = tk.Label(self.window, text="Emp Name", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empName.place(x=80, y=80)
        self.empNameTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empNameTxt.place(x=205, y=89)

        takeImg = tk.Button(self.window, text="Take Images", command=self.collectImage, fg="white", bg="#363e75", width=15,
                            height=2,
                            activebackground="#118ce1", font=('times', 15, ' bold '), cursor='hand2')
        takeImg.place(x=80, y=350)

        trainImg = tk.Button(self.window, text="Train Images", command=self.trainModel, fg="white", bg="#363e75", width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '), cursor='hand2')
        trainImg.place(x=350, y=350)

        predictImg = tk.Button(self.window, text="Predict", command=self.makePrediction, fg="white", bg="#363e75",
                             width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '), cursor='hand2')
        predictImg.place(x=600, y=350)

        pred_monitor_img = tk.Button(self.window, text="Monitor", command=self.monitorFace, fg='white', bg='#363e75',
                            width=15,
                            height=2,
                            activebackground="#118ce1", font=('times', 15, ' bold '), cursor='hand2')
        pred_monitor_img.place(x=80,y=510)

        quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="#363e75", width=10, height=2,
                               activebackground="#118ce1", font=('times', 15, 'bold'), cursor='hand2')
        quitWindow.place(x=650, y=510)

        lbl3 = tk.Label(self.window, text="Notification : ", width=15, fg="white", bg="#363e75", height=2,
                        font=('times', 15))
        self.message = tk.Label(self.window, text="", bg="white", fg="black", width=30, height=1,
                                activebackground="#e47911", font=('times', 15))
        self.message.place(x=220, y=220)
        lbl3.place(x=80, y=260)

        self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2, activebackground="#bbc7d4",
                           font=('times', 15))
        self.message.place(x=220, y=260)

        self.window.mainloop()

    def collectImage(self):
        empIDVal = self.empIDTxt.get()
        name = (self.empNameTxt.get())
        config_path = os.path.join('src','config','config.yaml')
        content = read_yaml(config_path)
        faces = content['base']['number_of_image']
        log_dir = content['base']['log_dir']
        log_file = content['base']['log_file']
        file = os.path.join('src',log_dir, log_file)

        args = argparse.ArgumentParser()
        args.add_argument('--params','--p', default = config_path)
        args.add_argument('--name','--n', default = name)
        parsed_args = args.parse_args()

        if name:
            log('data collection started', file)
            app = data_collect(parsed_args.params, parsed_args.name)
            app.open_webcam()
            log('data collection finished',file)
            log('stage 01 completed', file)
            notification = "We have collected " + str(faces) + " images for training."
            self.message.configure(text=notification)
        else:
            notification = 'Please enter your name.'
            self.message.configure(text=notification)



    def trainModel(self):
        config_path = os.path.join('src','config','config.yaml')
        content = read_yaml(config_path)
        log_dir = content['base']['log_dir']
        log_file = content['base']['log_file']
        file = os.path.join('src',log_dir, log_file)

        args = argparse.ArgumentParser()
        args.add_argument('--config','--c',default = config_path)
        args.add_argument('--params', '--p', default = 'src/params.yaml')
        parsed_args = args.parse_args()

        log('genarating pickle file', file)
        generate_pkl_file(parsed_args.config)
        log('stage 02 compleed',file)

        log('feature extraction started', file)
        feature_extractor(config_path=parsed_args.config, model_param_path=parsed_args.params)
        log('feature extraction completed', file)
        log('stage 03 completed', file)

        notification = "We have collected face embeddings"
        self.message.configure(text=notification)

    def makePrediction(self):
        config_path = os.path.join('src','config','config.yaml')
        content = read_yaml(config_path)
        log_dir = content['base']['log_dir']
        log_file = content['base']['log_file']
        file = os.path.join('src',log_dir, log_file)

        args = argparse.ArgumentParser()
        args.add_argument('--params','--p', default = config_path)
        parsed_args = args.parse_args()

        log('Its Monitoring time',file)
        app = prediction(parsed_args.params)
        app.openWebcam()

    def monitorFace(self):
        config_path = os.path.join('src','config','config.yaml')
        content = read_yaml(config_path)
        log_dir = content['base']['log_dir']
        log_file = content['base']['log_file']
        file = os.path.join('src',log_dir, log_file)

        args = argparse.ArgumentParser()
        args.add_argument('--params','--p', default = config_path)
        args.add_argument('--track','--t', default=True)
        parsed_args = args.parse_args()

        log('Its Monitoring time',file)
        app = prediction(parsed_args.params, parsed_args.track)
        app.openWebcam()

    def close_window(self):
        self.window.destroy()


if __name__ == '__main__':
    app = RegistrationModule()
