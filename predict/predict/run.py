import json
import argparse
import os
import time
from collections import OrderedDict

from keras.models import load_model
from numpy import argsort

from preprocessing.preprocessing.embeddings import embed

import logging

logger = logging.getLogger(__name__)


class TextPredictionModel:
    def __init__(self, model, params, labels_to_index):
        self.model = model
        self.params = params
        self.labels_to_index = labels_to_index
        self.labels_index_inv = {ind: lab for lab, ind in self.labels_to_index.items()}

    @classmethod
    def from_artefacts(cls, artefacts_path: str):
        """
            from training artefacts, returns a TextPredictionModel object
            :param artefacts_path: path to training artefacts
        """
        # TODO: CODE HERE
        #? TO CHECK
        # load model
        model = load_model(f'{artefacts_path}/model.h5')

        # TODO: CODE HERE
        #? TO CHECK
        # load params
        with open(f'{artefacts_path}/params.json') as json_file:
            params = json.load(json_file)

        # TODO: CODE HERE
        # load labels_to_index
        with open(f'{artefacts_path}/labels_index.json') as json_file:
            json_file_str = json_file.read()
            labels_to_index = json.loads(json_file_str)

        return cls(model, params, labels_to_index)

    def predict(self, text_list, top_k=5):
        """
            predict top_k tags for a list of texts
            :param text_list: list of text (questions from stackoverflow)
            :param top_k: number of top tags to predict

            return an array with arrays containing tag prediction value for each texts in input
        """
        tic = time.time()

        logger.info(f"Predicting text_list=`{text_list}`")

        # TODO: CODE HERE
        #? TO CHECK
        # embed text_list
        embedding = embed([text_list])
        # TODO: CODE HERE
        #? TO CHECK
        # predict tags indexes from embeddings
        tags_indexes = self.model.predict(embedding)
        # TODO: CODE HERE
        #? TO CHECK
        # from tags indexes compute top_k tags for each text
        top_k_labels = argsort(tags_indexes)[0][:top_k]
        predictions = [self.labels_to_index[str(index)] for index in top_k_labels]
        logger.info("Prediction done in {:2f}s".format(time.time() - tic))
        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts_path", help="path to trained model artefacts")
    parser.add_argument("text", type=str, default=None, help="text to predict")
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    model = TextPredictionModel.from_artefacts(args.artefacts_path)

    if args.text is None:
        while True:
            txt = input("Type the text you would like to tag: ")
            predictions = model.predict([txt])
            print(predictions)
    else:
        print(f'Predictions for `{args.text}`')
        print(model.predict([args.text]))
