import os
import shutil
import tempfile
import unittest
from datetime import datetime

from modelzoo.modelzoo import ModelZoo

TEST_MODELSCOPE_API = '44aa873f-4a8c-492c-a3b2-7ecd1d53f344'
TEST_OSS_CONFIG_FILE = '/mnt/workspace/workgroup/' + \
    'zhoulou/AIGC/modelzoo/oss/ossconfig'


class TestModelZoo(unittest.TestCase):
    def setUp(self):
        self.modelzoo = ModelZoo(modelscope_api=TEST_MODELSCOPE_API,
                                 modelzoo_name='test',
                                 oss_config_file=None)
        self.test_dir = tempfile.mkdtemp()

    def test_create_model(self):
        # create test files
        test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(test_dir, 'test_folder'))
        with open(os.path.join(test_dir, 'test.txt'), 'w') as f:
            f.write('test file')

        # test create model
        model_name = f"test_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_meta = self.modelzoo.create_model(test_dir,
                                                model_name=model_name,
                                                model_description='test model',
                                                model_tags=['test'])

        self.assertEqual(model_meta.model_name.text, model_name)
        self.assertEqual(model_meta.model_description.text, 'test model')
        self.assertIn('test', model_meta.model_tags.texts)
        self.assertIn('author_name_zhoulou', model_meta.model_tags.texts)
        self.assertIn('modelzoo_name_test', model_meta.model_tags.texts)

        # delete test files and model
        shutil.rmtree(test_dir)
        self.modelzoo.delete_model(model_name)

    def test_delete_model(self):
        # create test files and model
        test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(test_dir, 'test_folder'))
        with open(os.path.join(test_dir, 'test.txt'), 'w') as f:
            f.write('test file')

        model_name = f"test_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.modelzoo.create_model(test_dir,
                                   model_name=model_name,
                                   model_description='test model',
                                   model_tags=['test'])

        # test delete model
        message = self.modelzoo.delete_model(model_name)

        self.assertEqual(
            message,
            f"The model '{model_name}' has been successfully deleted.")
        self.assertIsNone(self.modelzoo.get_model_by_name(model_name))

        # delete test files
        shutil.rmtree(test_dir)

    def test_modify_model(self):
        # create test files
        test_dir = tempfile.mkdtemp()

        os.makedirs(os.path.join(test_dir, 'test_folder'))
        with open(os.path.join(test_dir, 'test.txt'), 'w') as f:
            f.write('test file')

        # create test model
        model_name = f"test_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.modelzoo.create_model(test_dir,
                                   model_name=model_name,
                                   model_description='test model',
                                   model_tags=['test'])

        # modify model
        new_description = 'updated model'
        new_version = '1.1.0'
        test_dir2 = tempfile.mkdtemp()
        os.makedirs(os.path.join(test_dir2, 'test_folder'))
        with open(os.path.join(test_dir2, 'test.txt'), 'w') as f:
            f.write('test_a file')
        message = self.modelzoo.modify_model(test_dir2,
                                             model_name=model_name,
                                             version=new_version,
                                             description=new_description)

        # check that the model was modified correctly
        self.assertEqual(
            message, f"The model '{model_name}' has been successfully ' + \
                'modified on ModelScope and updated in the local model zoo.")
        updated_model = self.modelzoo.get_model_by_name(model_name)
        # print(updated_model.chunks.texts)
        message = self.modelzoo.delete_model(model_name)

        self.assertEqual(updated_model.model_version.text, new_version)

        # clean up
        shutil.rmtree(test_dir)
        shutil.rmtree(test_dir2)

    def test_download_model(self):
        # create test files and model
        test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(test_dir, 'test_folder'))
        with open(os.path.join(test_dir, 'test.txt'), 'w') as f:
            f.write('test file')
        model_name = f"test_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.modelzoo.create_model(test_dir,
                                   model_name=model_name,
                                   model_description='test model',
                                   model_tags=['test'])

        # download model
        output_dir = tempfile.mkdtemp()
        self.modelzoo.download_model(model_name, output_dir)

        # check if downloaded files match original files
        downloaded_files = os.listdir(output_dir)
        self.assertGreater(len(downloaded_files), 0)
        for f in downloaded_files:
            self.assertTrue(os.path.exists(os.path.join(output_dir, f)))

        with open(
                os.path.join(output_dir, self.modelzoo.meta.author_name,
                             model_name, 'test.txt'), 'r') as f:
            content = f.read()
        self.assertEqual(content, 'test file')

        self.modelzoo.delete_model(model_name)

        # delete test files and model
        shutil.rmtree(test_dir)
        shutil.rmtree(output_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()
