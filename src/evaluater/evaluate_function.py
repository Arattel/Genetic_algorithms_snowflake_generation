import os
import glob
import json
import argparse
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def evaluate(base_model_name, weights_file, image_source, img_format='jpg'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions = predict(nima.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    return samples


class HelperEvaluator:
    def __init__(self, base_model_name, weights_file):
        self.nima = Nima(base_model_name, weights=None)
        self.nima.build()
        self.nima.nima_model.load_weights(weights_file)

    def evaluate(self, image_dir, img_format='jpg'):
        samples = image_dir_to_json(image_dir, img_type='jpg')
        data_generator = TestDataGenerator(samples, image_dir, 64, 10, self.nima.preprocessing_function(),
                                           img_format=img_format)

        # get predictions
        predictions = predict(self.nima.nima_model, data_generator)

        # calc mean scores and add to samples
        for i, sample in enumerate(samples):
            sample['mean_score_prediction'] = calc_mean_score(predictions[i])

        return samples


if __name__ == '__main__':
    print('EVALUATING')
    print('*' * 38)
    parser = argparse.ArgumentParser()
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)

    args = parser.parse_args()

    # Basic config
    WORKDIR: str = os.getcwd()
    BASE_MODEL_NAME: str = 'MobileNet'
    WEIGHTS_FILE: str = 'src/weights.hdf5'
    print(WEIGHTS_FILE)
    print(f'PWD: {os.listdir("..")}')

    he = HelperEvaluator(base_model_name=BASE_MODEL_NAME, weights_file=WEIGHTS_FILE)
    print(he.evaluate(args.image_source))
