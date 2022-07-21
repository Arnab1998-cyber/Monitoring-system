from utils.all_utils import read_yaml, create_directory, log
import argparse
import os
import pickle
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import keras_applications
import numpy as np
from tqdm import tqdm

def extractor(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result=model.predict(preprocessed_img).flatten()
    return result

def feature_extractor(config_path, model_param_path):
    content = read_yaml(config_path)
    params = read_yaml(model_param_path)
    base = content['base']
    artifacts_dir = base['artifacts_dir']
    pickle_format_data_dir = base['pickle_format_data_dir']
    img_pickle_file_name = base['img_pickle_file_name']
    model_dir = base['model_dir']
    model_file = base['model_file']
    log_dir = base['log_dir']
    log_file = base['log_file']
    log_path = os.path.join(log_dir, log_file)

    img_pickle_file = os.path.join('src',artifacts_dir, pickle_format_data_dir, img_pickle_file_name)
    file_names=pickle.load(open(img_pickle_file, 'rb'))
    model_path = os.path.join('src',model_dir,model_file)


    model_name=params['base']['base_model']
    include_top=params['base']['include_top']
    pooling=params['base']['pooling']

    model=VGGFace(include_top=include_top,model=model_name,pooling=pooling, input_shape=(224,224,3))

    feature_extrct_dir = base['feature_extraction_dir']
    extracted_features_name = base['extracted_features_name']
    feature_extraction_path = os.path.join('src',artifacts_dir,feature_extrct_dir)
    create_directory([feature_extraction_path])
    feature_name=os.path.join(feature_extraction_path,extracted_features_name)

    features = []
    for file in tqdm(file_names):
        features.append(extractor(file, model))
    pickle.dump(features, open(feature_name, 'wb'))
    log('feature file is ready', log_path)
    #pickle.dump(model,open(model_path,'wb'))
    model.save(model_path)
    log('model file is ready', log_path)



if __name__ == '__main__':
    config_path = os.path.join('src','config','config.yaml')
    content = read_yaml(config_path)
    log_dir = content['base']['log_dir']
    log_file = content['base']['log_file']
    file = os.path.join('src',log_dir, log_file)

    args = argparse.ArgumentParser()
    args.add_argument('--config','--c',default = config_path)
    args.add_argument('--params', '--p', default = 'src/params.yaml')
    parsed_args = args.parse_args()

    log('feature extraction started', file)
    feature_extractor(config_path=parsed_args.config, params_path=parsed_args.params)
    log('feature extraction completed', file)
    log('stage 03 completed', file)
