import unittest

from PIL import Image

from modelzoo.prompt import Prompt


class TestPrompt(unittest.TestCase):
    def test_convertFromImage(self):
        # Construct an Image object containing the prompt information
        exif = 'This is a prompt.\nNegative prompt: this is ' + \
            'a negative prompt\nSteps: 26, Sampler: Euler a, ' + \
            'CFG scale: 6.5, Seed: 1791574510, Size: 768x1024, ' + \
            'Model hash: 9aba26abdf, Model: Deliberate, ENSD: 31337'
        image = Image.new(mode='RGB', size=(100, 100))
        image.info['parameters'] = exif

        # Construct the expected Prompt object
        prompt = Prompt(
            prompt='This is a prompt.',
            sampler='Euler a',
            model='Deliberate',
            seed=1791574510,
            cfg_scale=6.5,
            steps=26,
            negative_prompt='this is a negative prompt',
        )

        # Call the method to get the actual Prompt object
        actual_prompt = Prompt.convertFromImage(image)

        # Use an assertion statement to
        # compare the expected output to the actual output
        assert actual_prompt == prompt


if __name__ == '__main__':
    unittest.main()
