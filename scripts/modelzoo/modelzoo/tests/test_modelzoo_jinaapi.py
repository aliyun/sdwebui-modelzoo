import os
import shutil
import tempfile
import unittest

from modelzoo.modelzoo import ModelMeta, ModelZoo

TEST_MODELSCOPE_API = '3ceb1149-f13a-42f0-bfaa-4932eca2a9a5'
TEST_OSS_CONFIG_FILE = None


class TestModelZooJINA(unittest.TestCase):
    def setUp(self):
        self.mz = ModelZoo(modelscope_api=TEST_MODELSCOPE_API,
                           modelzoo_name='test_jina',
                           oss_config_file=None)
        self.test_dir = tempfile.mkdtemp()

    def test_list_model(self):
        # create model
        model_name = 'list_model'
        self.create_test_model(model_name)

        # call the list_model method
        model_names = self.mz.list_model()
        # check the returned list
        self.assertIn(model_name, model_names)

        # delete the test model
        self.delete_test_model(model_name)

    def test_get_model_by_name(self):
        # Add a model to the model zoo
        model_name = 'get_model_by_name'
        self.create_test_model(model_name)

        # Test getting an existing model
        result = self.mz.get_model_by_name(model_name)
        self.assertIsNotNone(result)
        self.assertEqual(ModelMeta(result).model_name, model_name)

        # Test getting a non-existing model
        non_exists_model_name = 'test_model_non_exists'
        result = self.mz.get_model_by_name(non_exists_model_name)
        self.assertIsNone(result)

        # delete the test model
        self.delete_test_model(model_name)

    def test_get_model_by_tag(self):
        model_name_1 = 'get_model_by_tag_1'
        model_name_2 = 'get_model_by_tag_2'
        model_tags_1 = ['tag1', 'tag2']
        model_tags_2 = ['tag2', 'tag3']
        self.create_test_model(model_name_1, model_tags=model_tags_1)
        self.create_test_model(model_name_2, model_tags=model_tags_2)
        # Test getting a model by a tag that exists
        models = self.mz.get_model_by_tag('tag2')
        self.assertEqual(len(models), 2)
        for model in models:
            self.assertIn(
                ModelMeta(model).model_name, [model_name_1, model_name_2])

        # Test getting a model by a tag that does not exist
        non_exists_model_tag = 'test_model_tag_non_exists'
        result = self.mz.get_model_by_tag(non_exists_model_tag)
        self.assertIsNone(result)

        # delete the test model
        self.delete_test_model(model_name_1)
        self.delete_test_model(model_name_2)

    def test_search_model_by_name(self):
        # create model
        model_name_1 = 'search_model_by_name_1'
        model_name_2 = 'search_model_by_name_2'
        model_name_3 = 'similarity'
        test_topk = 4
        self.create_test_model(model_name_1)
        self.create_test_model(model_name_2)
        self.create_test_model(model_name_3)

        models = self.mz.search_model_by_name(model_name_1, topk=test_topk)
        self.assertEqual(len(models), test_topk)
        # Ensure the correctness of the returned order and the similarity.
        self.assertEqual(ModelMeta(models[0]).model_name, model_name_1)
        self.assertEqual(ModelMeta(models[1]).model_name, model_name_2)

        # delete model
        self.delete_test_model(model_name_1)
        self.delete_test_model(model_name_2)
        self.delete_test_model(model_name_3)

    def test_search_model_by_tag(self):
        # create model
        model_name_1 = 'search_model_by_tag_1'
        model_name_2 = 'search_model_by_tag_2'
        model_name_3 = 'search_model_by_tag_3'
        model_tags_1 = ['1_test', '2_test']
        model_tags_2 = ['3_test', '4_test']
        model_tags_3 = ['similarity_', 'similarity__']
        test_topk = 2
        self.create_test_model(model_name_1, model_tags=model_tags_1)
        self.create_test_model(model_name_2, model_tags=model_tags_2)
        self.create_test_model(model_name_3, model_tags=model_tags_3)

        models = self.mz.search_model_by_tag(model_tags_1[1], topk=test_topk)
        self.assertEqual(len(models), test_topk)

        # Ensure the correctness of the returned order and the similarity.
        self.assertEqual(ModelMeta(models[0]).model_name, model_name_1)
        self.assertEqual(ModelMeta(models[1]).model_name, model_name_2)

        # delete model
        self.delete_test_model(model_name_1)
        self.delete_test_model(model_name_2)
        self.delete_test_model(model_name_3)

    def test__call__(self):
        # create model
        model_name_1 = 'test_call_1'
        self.create_test_model(model_name_1)

        model_zoo = self.mz()
        # Assert
        self.assertEqual(model_zoo[-1].model_name.text, model_name_1)

        # delete model
        self.delete_test_model(model_name_1)

    def create_test_model(self, model_name, **kwargs):
        # create test folder
        if not os.path.exists(os.path.join(self.test_dir, 'test_folder')):
            os.makedirs(os.path.join(self.test_dir, 'test_folder'))
        with open(os.path.join(self.test_dir, 'test.txt'), 'w') as f:
            f.write('test file')
        # To create a model so that it appears in the model list.
        self.mz.create_model(model_dir=self.test_dir,
                             model_name=model_name,
                             **kwargs)

    def delete_test_model(self, model_name):
        self.mz.delete_model(model_name=model_name)

    def tearDown(self):
        shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()
