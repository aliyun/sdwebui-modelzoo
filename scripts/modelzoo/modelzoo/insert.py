import logging
from typing import List, Union

from docarray import Document
from docarray.typing import Image

from .metainfo import ModelMeta, ModelMetaIdentity
from .prompt import Prompt
from .utils import split_prompt

# Insert with DocumentArray is hard, we make some function here to support
# insert_history_records:
# inserting history records to the given model_meta instance
# insert_tags:
# inserting tags to the given model_meta instance


def insert_history_records(model_meta: Union[ModelMeta, Document],
                           prompt: Union[Prompt, str],
                           output_image: Image = None,
                           **kwargs) -> None:
    """Insert a history record into a model's history records.

    Args:
        model_meta (Union[ModelMeta, Document]):
        A model's metadata or document.
        prompt (Union[Prompt, str]):
            A prompt or string to be added to the history records.
        output_image (Image, optional):
            An image associated with the prompt. Defaults to None
            (If the prompt parameter is of type str, it cannot be None.).
        **kwargs: Additional keyword arguments for the prompt.

    Returns:
        Union[ModelMeta, Document]: The modified model metadata or document.

    Examples:
        model_meta = ModelMeta(
            model_name='my_model',
            model_description='my_description')

        prompt = 'this is a test prompt of str'
        output_image = Image.open("test.png")
        insert_history_records(model_meta, prompt, output_image)
        for p in model_meta.history_records:
            print(
                "prompt: ", Prompt(p).prompt, "\toutput_image: ",
                Prompt(p).output_image)
        More information please refer to
        tests/test_modelmeta#test_insert_history_records()
    """
    prompt_kwargs = None
    if isinstance(prompt, str):
        if output_image is None:
            raise ValueError(
                'insert_history_records() missing required argument: "prompt".'
                + ' You can use this method in the following format: '
                + 'insert_history_records(prompt=your_prompt, '
                + 'output_image=your_image_url, other Prompt keywargs)')
        prompt_kwargs = {'output_image': output_image}

        prompt_kwargs['sampler'] = kwargs.get(
            'sampler',
            getattr(model_meta, 'model_default_sampler',
                    'model_default_sampler_test'))
        prompt_kwargs['model'] = kwargs.get(
            'model', getattr(model_meta, 'model_name', 'default_str'))
        prompt_kwargs['cfg_scale'] = kwargs.get(
            'cfg_scale', getattr(model_meta, 'model_default_cfg_scale', '0.7'))
        prompt_kwargs['steps'] = kwargs.get(
            'steps', getattr(model_meta, 'model_default_steps', '20'))
        prompt_kwargs['negative_prompt'] = kwargs.get('negative_prompt', '')
        prompt_kwargs['seed'] = kwargs.get('seed', '')
        prompt_kwargs['prompt_keys'] = kwargs.get('prompt_keys',
                                                  split_prompt(prompt, ' '))
        prompt_kwargs['negative_prompt_keys'] = kwargs.get(
            'negative_prompt_keys',
            split_prompt(prompt_kwargs['negative_prompt']))
        prompt = Prompt(prompt, **prompt_kwargs)

    if isinstance(model_meta, Document):
        # according to define of Prompt and ModelMeta:
        # if model_meta = Document(ModelMeta) and
        # refer to https://docarray.jina.ai/fundamentals/documentarray/
        #   access-elements/#index-by-nested-structure
        # model_meta.chunks[-1].text = ModelMeta.identity.text,
        #   which  is 'ModelMeta'
        if not hasattr(model_meta, 'chunks'):
            raise ValueError(
                'Input model_meta as Document(ModelMeta) instance' +
                ' should have chunks')
        if len(model_meta.chunks) == 0:
            raise ValueError(
                'Input model_meta as Document(ModelMeta) instance' +
                ' should have chunk length > 0')
        if model_meta.chunks[-1].text != ModelMetaIdentity:
            raise ValueError(
                'Input model_meta as Document(ModelMeta) ' +
                'instance should have chunks with identity ends ' +
                'as \'ModelMeta\', which means this input model_meta ' +
                'is not a Document(ModelMeta) instance')

        model_meta.history_records.extend([Document(prompt)])
    elif isinstance(model_meta, ModelMeta):
        model_meta.history_records.extend([Document(prompt)])
    else:
        raise ValueError(
            'The input model_meta should be either ' +
            'a Document(ModelMeta) instance or a ModelMeta instance.')

    if prompt_kwargs is not None:
        res = 'insert_history_records : ' + \
            'Using default parameters to create Prompt Instance: {'
        for param, value in prompt_kwargs.items():
            if param not in kwargs.keys():
                res += f'{param}: {value}, '
        logging.info(res + '}')
    return model_meta


def insert_tags(model_meta: Union[ModelMeta, Document],
                tags: Union[List[str], str]) -> Union[ModelMeta, Document]:
    """
    Inserts tags to the given `model_meta`.
        If `tags` is a string, it is converted to a list.

    Args:
        model_meta: Either a `Document` or a `ModelMeta` instance
            to which the tags will be inserted.
        tags: A list of tags or a string containing a single tag.

    Returns:
        The modified `model_meta` instance with the inserted tags.

    Raises:
        ValueError: If the `model_meta` instance is
            neither a `Document` nor a `ModelMeta` instance.

    Examples:
        model_meta = ModelMeta(
            model_name='my_model',
            model_description='my_description',
            model_tags=['anime']
        )
        print("before insert_tags: ", model_meta.model_tags)
        insert_tags(model_meta, ["realistic", "3D"])
        print("after insert_tags: ", model_meta.model_tags)
        More information can refer to
            tests/test_modelmeta.py#test_insert_tags()
    """
    if isinstance(tags, str):
        tags = [tags]

    if isinstance(model_meta, Document):
        # Assuming that there's at least one tag already in the document
        t0 = model_meta.model_tags[0]
        model_meta.model_tags.extend([
            Document(text=tag,
                     parent_id=t0.parent_id,
                     granularity=t0.granularity,
                     modality=t0.modality) for tag in tags
        ])
    elif isinstance(model_meta, ModelMeta):
        model_meta.model_tags = list(model_meta.model_tags)
        model_meta.model_tags.extend(tags)
    else:
        raise ValueError(
            'The input model_meta should be either a Document(ModelMeta)' +
            ' instance or a ModelMeta instance.')

    return model_meta
