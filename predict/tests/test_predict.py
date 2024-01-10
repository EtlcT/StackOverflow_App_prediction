import unittest
import pandas as pd
from unittest.mock import MagicMock
from keras.models import load_model
import tempfile
from preprocessing.preprocessing import utils
from predict.predict import run as runPredict
from train.tests import test_model_train
from train.train import run as runTrain


class TestPredict(unittest.TestCase):
    def test_predictLabels(self):
        """
        test if predict function correctly output labels from the prediction of the model
        """
        params = {
            'batch_size': 2,
            'epochs': 1,
            'dense_dim': 64,
            'min_samples_per_label': 2,
            'verbose': 1
        }
        utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value= test_model_train.load_dataset_mock())

        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            runTrain.train(dataset_path='fake_path', train_conf=params, model_path=model_dir, add_timestamp=False)
            model = runPredict.TextPredictionModel.from_artefacts(model_dir)
            title_to_predict = ['Is it possible to execute the procedure of a function in the scope of the caller?']
            self.assertListEqual(sorted(model.predict(text_list=title_to_predict,top_k=2)), ['php', 'ruby-on-rails'])
            
