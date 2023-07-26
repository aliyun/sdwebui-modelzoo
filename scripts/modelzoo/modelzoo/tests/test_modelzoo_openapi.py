import os
import sys
import unittest

import torch
from diffusers import StableDiffusionPipeline

from modelzoo.modelzoo import ModelZoo

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))

TEST_MODELSCOPE_API = '3ceb1149-f13a-42f0-bfaa-4932eca2a9a5'
TEST_OSS_CONFIG_FILE = None


class TestModelZooOpenapi(unittest.TestCase):
    def setUp(self):
        self.modelzoo = ModelZoo(modelscope_api=TEST_MODELSCOPE_API,
                                 modelzoo_name='test_openapi',
                                 oss_config_file=TEST_OSS_CONFIG_FILE)

    def test_insert_image(self):
        model_id = 'runwayml/stable-diffusion-v1-5'
        model_name = 'midjourney_large-finetune'
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16)
        pipe = pipe.to('cuda')

        # For assert
        model = self.modelzoo.get_model_by_name(model_name)
        length = len(model.history_records)

        # Configed parameters
        prompt = 'a photo of an astronaut riding a horse on mars'
        negative_prompt = 'less figures'
        seed = 12341351213
        guidance_scale = 4
        # Pipeline parameters
        num_images_per_prompt = 3
        # In principle, you can play any parameter of the diffusers pipeline.
        self.modelzoo.insert_image(pipe,
                                   model_name=model_name,
                                   oss_dir='wzh-zhoulou/aigc/models/',
                                   prompt=prompt,
                                   negative_prompt=negative_prompt,
                                   seed=seed,
                                   guidance_scale=guidance_scale,
                                   num_images_per_prompt=num_images_per_prompt)

        # Assert
        self.assertEqual(length + num_images_per_prompt,
                         len(model.history_records))


if __name__ == '__main__':
    unittest.main()
