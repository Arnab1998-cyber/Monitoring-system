from setuptools import setup

setup(
    name='src',
    version='0.0.1',
    author='Arnab Mitra',
    description='Face Monitering System',
    packages=['src'],
    install_requires=['tensorflow','keras','opencv-python','keras-vggface','cmake','imutils',
                    'dlib','Keras-Applications','sklearn','tqdm','pyttsx3']
)