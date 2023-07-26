import unittest

from modelzoo.utils import split_prompt


class TestSplitPrompt(unittest.TestCase):
    def test_comma_delimiter(self):
        prompt_str = 'prompt1, prompt2, prompt3'
        expected_output = ['prompt1', 'prompt2', 'prompt3']
        self.assertEqual(split_prompt(prompt_str), expected_output)

    def test_pipe_delimiter(self):
        prompt_str = 'prompt1|prompt2|prompt3'
        expected_output = ['prompt1', 'prompt2', 'prompt3']
        self.assertEqual(split_prompt(prompt_str, '|'), expected_output)

    def test_space_delimiter(self):
        prompt_str = 'prompt1 prompt2 prompt3'
        expected_output = ['prompt1', 'prompt2', 'prompt3']
        self.assertEqual(split_prompt(prompt_str, ' '), expected_output)

    def test_no_delimiter(self):
        prompt_str = 'prompt1'
        expected_output = ['prompt1']
        self.assertEqual(split_prompt(prompt_str), expected_output)


if __name__ == '__main__':
    unittest.main()
