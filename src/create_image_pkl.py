import pickle
import os
from utils.all_utils import log, read_yaml, create_directory
import argparse

def generate_pkl_file(config_path):
    content = read_yaml(config_path)
    base = content['base']
    artifacts_dir = base['artifacts_dir']
    pickle_format_data_dir = base['pickle_format_data_dir']
    img_pickle_file_name = base['img_pickle_file_name']
    data_path = base['data_dir']
    log_dir = base['log_dir']
    log_path = base['log_file']
    log_file = os.path.join(log_dir,log_path)

    raw_local_dir_path = os.path.join('src',artifacts_dir, pickle_format_data_dir)
    create_directory(dirs=[raw_local_dir_path])
    pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)
    
    persons = os.listdir(data_path)
    file_names = []
    for person in persons:
        for file in os.listdir(os.path.join(data_path,person)):
            file_names.append(os.path.join(data_path, person, file))

    log('pickle file for images created', log_file)
    log('There are {} persons'.format(os.listdir(data_path)), log_file)
    with open(pickle_file, 'wb') as f:
        pickle.dump(file_names, f)


if __name__ == '__main__':
    config_path = os.path.join('src','config','config.yaml')
    args = argparse.ArgumentParser()
    args.add_argument('--config','--c',default = config_path)
    parsed_args = args.parse_args()

    content = read_yaml(config_path)
    log_dir = content['base']['log_dir']
    log_file = content['base']['log_file']
    file = os.path.join('src',log_dir, log_file)

    log('genarating pickle file', file)
    generate_pkl_file(parsed_args.config)
    log('stage 02 compleed')

