import unittest

from docarray import Document

from modelzoo.insert import insert_history_records, insert_tags
from modelzoo.metainfo import ModelMeta
from modelzoo.prompt import Prompt


class TestModelMeta(unittest.TestCase):
    def setUp(self):
        return

    def test_insert_history_records(self):
        model_meta = ModelMeta(model_name='test_model',
                               model_description='test_description',
                               model_tags=['tag1', 'tag2'])

        prompt_str1 = 'test prompt 1'
        prompt_str2 = 'test prompt 2'
        prompt_str3 = 'test prompt 3'
        image_url = 'http://pai-vision-exp.oss-cn-zhangjiakou.aliyuncs.com' + \
            '/wzh-zhoulou/metric_learning/230303/test.png'
        prompt1 = Prompt(prompt_str1, image_url)
        prompt2 = Prompt(prompt_str2,
                         image_url,
                         negative_prompt='test negative prompt',
                         sampler='test_sampler')

        self.assertEqual(len(model_meta.history_records), 2)

        insert_history_records(model_meta, prompt1)
        self.assertEqual(len(model_meta.history_records), 3)

        insert_history_records(model_meta, prompt2)
        self.assertEqual(len(model_meta.history_records), 4)

        insert_history_records(model_meta, prompt_str3, image_url)
        self.assertEqual(len(model_meta.history_records), 5)

    def test_insert_tags(self):

        # each model should start with at least 2 tags and 2 historys
        model_meta = ModelMeta(model_name='test',
                               model_description='test_description',
                               model_tags=['SD2.1', 'AIGC'])
        print(model_meta.model_tags)

        itags = ['test_tag1', 'test_tag2']
        insert_tags(model_meta, itags)
        print(model_meta.model_tags)
        model_meta = Document(model_meta)

        itags = ['test_tag3', 'test_tag4']
        insert_tags(model_meta, itags)
        self.assertEqual(model_meta.model_tags.texts, [
            'SD2.1', 'AIGC', 'test_tag1', 'test_tag2', 'test_tag3', 'test_tag4'
        ])


if __name__ == '__main__':
    unittest.main()
