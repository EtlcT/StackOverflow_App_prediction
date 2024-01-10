import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils



class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        #? DONE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value = 100)
        self.assertEqual(base._get_num_train_batches(),4)

    def test__get_num_test_batches(self):
        # TODO: CODE HERE
        #? DONE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value = 100)
        self.assertEqual(base._get_num_test_batches(),1)
    
    def test_get_index_to_label_map(self):
        # TODO: CODE HERE
        #? DONE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value = ['java','php','python'])
        self.assertEqual(base.get_index_to_label_map(),{'java':0,'php':1,'python':2})
    
    def test_get_index_to_label_map_no_duplicate(self):
        # TODO: CODE HERE
        #? DONE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value = ['java','php','python','php'])
        self.assertEqual(base.get_index_to_label_map(),{'java':0,'php':1,'python':2})
    
    def test_index_to_label_and_label_to_index_are_identity(self):
        # TODO: CODE HERE
        #? DONE
        base = utils.BaseTextCategorizationDataset(20,0.8)
        base._get_label_list = MagicMock(return_value = ['java','php','python','php'])
        index_to_label = base.get_index_to_label_map() # {'java':0,'php':1,'python':2}
        label_to_index = base.get_label_to_index_map() # {0:'java',1:'php',2:'python'}
        for key, value in index_to_label.items() : # the key in index_to_label are the value of the corresponding key in label_to_index
            self.assertEqual(key,label_to_index[value])
    
    def test_to_indexes(self):
        # TODO: CODE HERE
        #? DONE
        base = utils.BaseTextCategorizationDataset(20,0.8)
        base._get_label_list = MagicMock(return_value = ['java','php','python','php'])
        labels = ['java','php','python','php']
        self.assertEqual(base.to_indexes(labels),[0,1,2,1])

class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # TODO: CODE HERE
        #? DONE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3','id_4'],
            'tag_name': ['tag_a', 'tag_a','tag_b','tag_b'],
            'tag_id': [1, 2, 3, 4],
            'tag_position': [0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4']
        }))
        local = utils.LocalTextCategorizationDataset('fake_path',batch_size=1 ,train_ratio=0.5, min_samples_per_label=1 )
        self.assertEqual(local._get_num_samples(),4)

    
    def test_get_train_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        #? DONE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3','id_4'],
            'tag_name': ['tag_a', 'tag_a','tag_b','tag_b'],
            'tag_id': [1, 2, 3, 4],
            'tag_position': [0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4']
        }))
        local = utils.LocalTextCategorizationDataset('fake_path',batch_size=2 ,train_ratio=0.5, min_samples_per_label=1 )
        x_train_batch, y_train_batch = local.get_train_batch()
        self.assertEqual(x_train_batch.shape[0],2)
        self.assertEqual(y_train_batch.shape[0],2)
    
    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        #? DONE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3','id_4'],
            'tag_name': ['tag_a', 'tag_a','tag_b','tag_b'],
            'tag_id': [1, 2, 3, 4],
            'tag_position': [0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4']
        }))
        local = utils.LocalTextCategorizationDataset('fake_path',batch_size=2 ,train_ratio=0.5, min_samples_per_label=1 )
        x_test_batch, y_test_batch = local.get_test_batch()
        self.assertEqual(x_test_batch.shape[0],2)
        self.assertEqual(y_test_batch.shape[0],2)
    
    def test_get_train_batch_raises_assertion_error(self):
        # TODO: CODE HERE
        #? DONE
        #* will raise an error because there will be only 1 element in train set
        #* and we try to make batch size of 2 element so integer_floor(0.5) > 0
        #* will raise the error
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 0],
            'title': ['title_1', 'title_2']
        }))
        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset('fake_path',batch_size=2 ,train_ratio=0.5, min_samples_per_label=1)




