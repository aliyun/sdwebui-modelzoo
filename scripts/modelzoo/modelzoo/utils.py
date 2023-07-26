import base64
import hashlib
import os
import random
import shutil
from typing import List


class SafeStr:
    def __init__(self, secret: str):
        self.str = base64.b64encode(secret.encode()).decode()

    def __repr__(self):
        return f'({self.str})'

    def __call__(self):
        return base64.b64decode(self.str.encode()).decode()


def split_prompt(prompt_str: str, delimiter: str = ',') -> List[str]:
    """
    Splits a prompt string into individual prompt strings
        using the specified delimiter.

    Args:
        prompt_str: A string containing multiple prompts separated
            by the specified delimiter.
        delimiter: The delimiter character used to separate prompts
            in the prompt string. Default is ','.

    Returns:
        A list of individual prompt strings.
    """

    # Remove trailing delimiter from prompt string
    prompts = prompt_str.split(delimiter)

    # Remove whitespace from beginning and end of each prompt string
    prompts = [p.strip() for p in prompts if len(p.strip()) > 0]

    # Return list of individual prompt strings
    return prompts


def copy_tree(src, dst):
    """
    Recursively copy all files under the directory to the dest directory.

    Args:
        src (str): The path of the source directory to be copied.
        dst (str): The path of the destination directory.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory '{src}' does not exist.")

    if not os.path.exists(dst):
        os.makedirs(dst)

    if os.path.isdir(src):
        files = os.listdir(src)
        for f in files:
            copy_tree(os.path.join(src, f), os.path.join(dst, f))
    else:
        shutil.copy(src, dst)


def split_common_info(info: str):
    """
    Example:
        input: "
            Steps: 20,
            Sampler: DPM++ 2M Karras,
            CFG scale: 0.5,
            Seed: 1234,
            Size: 512,
            Model hash: abcdefg,
            Model: gpt2-medium
        "
        output: (
            '20',
            'DPM++ 2M Karras',
            '0.5',
            '1234',
            '512',
            'abcdefg',
            'gpt2-medium',
            []
        )

    Parses a comma-separated string containing configuration information
        and returns the parsed values.

    Args:
        info (str): A comma-separated string containing information.

    Returns:
        A tuple containing the parsed values of the configuration information:
            steps (int),
            sampler (str),
            cfg_scale (float),
            seed (int),
            size (int),
            model (str),
            and any additional items that were present in the input string.
    """
    # Split the input string into individual components
    steps, sampler, cfg_scale, seed, size, *items = info.split(', ')

    # Parse the individual components
    step_template = 'Steps: '
    steps = steps[len(step_template):]
    sampler_template = 'Sampler: '
    sampler = sampler[len(sampler_template):]
    cfg_template = 'CFG scale: '
    cfg_scale = cfg_scale[len(cfg_template):]
    seed_template = 'Seed: '
    seed = seed[len(seed_template):]
    size_template = 'Size: '
    size = size[len(size_template):]

    model_template = 'Model: '
    model = ''
    for item in items:
        if model_template in item:
            model = item[len(model_template):]
            break

    # Return the parsed values
    return steps, sampler, cfg_scale, seed, size, model, items


def upload_image2oss(bucket, image, oss_dir, model_name):
    filename = os.path.join(oss_dir, model_name, random_filename())
    bucket.put_object_from_file(filename, image)

    return filename


def random_filename(prefix=''):
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    suffix = '.png'
    filename = prefix + ''.join(random.choices(chars, k=10)) + suffix

    return filename


def get_file_md5(fname):
    m = hashlib.md5()
    with open(fname, 'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)

    return m.hexdigest()


DEFAULT_MODELZOO_FILE = 'https://pai-vision-data-sh.oss-{}' + \
    '.aliyuncs.com/aigc-data/modelzoo/modelzoo.bin'
DEFAULT_REGION_MODEL = 'https://pai-vision-data-sh.oss-{}.' + \
    'aliyuncs.com/aigc-data/'
