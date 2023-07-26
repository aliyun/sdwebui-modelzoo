from typing import List

from docarray import dataclass
from docarray.typing import Image, Text
from PIL import Image as PImage

from .utils import split_common_info

PromptIdentity = 'Prompt'


@dataclass
class Prompt:
    """
    Prompt represents a prompt used to train a machine learning model.

    Attributes:
        prompt (str): The prompt text.
        output_image (Image): The output image associated with the prompt.
        negative_prompt (str): The negative prompt text.
        sampler (str): The sampler used for the prompt.
        model (str): The model associated with the prompt.
        seed (str): The seed value used for the prompt.
        cfg_scale (float): The config scale for the prompt.
        steps (int): The number of steps for the prompt.
        prompt_keys (List[str]): A list of individual prompt strings
            derived from the original prompt text.
        negative_prompt_keys (List[str]): A list of individual prompt strings
            derived from the negative prompt text.
        identity (str): The identity of the prompt.
    """
    prompt: Text
    output_image: Image = None
    negative_prompt: Text = ''
    sampler: Text = ''
    model: Text = ''
    seed: Text = ''
    cfg_scale: float = 0.7
    steps: int = 20
    prompt_keys: List[Text] = ()
    negative_prompt_keys: List[Text] = ()
    identity: Text = PromptIdentity

    @staticmethod
    def convertFromImage(image: PImage.Image):
        """
        Converts EXIF metadata in an image file to a prompt object.

        Args:
            image (PIL.Image.Image):
                A PIL Image object containing EXIF metadata.

        Returns:
            A Prompt object containing the parsed prompt information
            from the EXIF metadata.

        """
        # Get the EXIF metadata from the image
        items = image.info or {}
        # exif = items.pop('parameters', None)
        exif = items.get('parameters', None)
        # Split the metadata into individual components
        # and parse the common information
        splitted_exif = exif.split('\n')
        common_info = splitted_exif[-1]
        steps, sampler, cfg_scale, seed, size, model, items = \
            split_common_info(common_info)

        # Create a Prompt object with the parsed information
        prompt = Prompt(
            prompt='',
            sampler=sampler,
            model=model,
            seed=seed,
            cfg_scale=(float)(cfg_scale),
            steps=(int)(steps),
        )

        # Extract the negative prompt from the EXIF metadata, if present
        negative_template = 'Negative prompt: '

        if len(splitted_exif) == 2:
            if 'Negative prompt: ' in splitted_exif[0]:
                prompt.negative_prompt = splitted_exif[0][len(negative_template
                                                              ):]
            else:
                prompt.prompt = splitted_exif[0]
        if len(splitted_exif) == 3:
            prompt.prompt = splitted_exif[0]
            prompt.negative_prompt = splitted_exif[1][len(negative_template):]

        return exif, prompt
