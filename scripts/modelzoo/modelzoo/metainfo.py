import logging
import os
from datetime import datetime
from typing import List, Optional

from docarray import Document, DocumentArray, dataclass
from docarray.typing import Text

from .prompt import Prompt

ModelMetaIdentity = 'ModelMeta'


@dataclass
class ModelMeta:
    """
    ModelMeta represents the metadata associated with a machine learning model.

    Attributes:
        model_name (str): The name of the model.
        model_description (str): A description of the model.
        model_version (str):
            The version of the model (default: 'model_version_test').
        model_author (str):
            The author of the model (default: 'model_author_test').
        model_default_sampler (str):
            The default sampler used for the model
                (default: 'model_default_sampler_test').
        model_default_cfg_scale (str):
            The default config scale for the model (default: '0.7').
        model_default_steps (str):
            The default number of steps for the model (default: '20').
        model_tags (List[str]):
            A list of tags associated with the model (default: []).
        history_records (List[Prompt]):
        A list of prompts used to train the model
            (default: [Document(Prompt(prompt='dummy'))]).
        identity (str): The identity of the model (default: 'ModelMeta').
    """
    model_name: Text
    model_description: Text = ''
    model_created: Text = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_updated: Text = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_version: Text = 'model_version_test'
    model_author: Text = 'model_author_test'
    model_dataset: Text = 'model_dataset_default'
    model_default_sampler: Text = 'model_default_sampler_test'
    model_default_cfg_scale: Text = '0.7'
    model_default_steps: Text = '20'
    # here we need 2 prompt as initialize,
    # other wise ,cast ModelMeta with Document(ModelMeta)
    #   will change this attr's  type to Document
    # this feature needs to fixed,
    # other wise every model will have 2 tags and 2 history_records in default
    model_file_list: List[Text] = DocumentArray([Document(), Document()])
    model_tags: List[Text] = DocumentArray([Document(), Document()])
    history_records: List[Prompt] = DocumentArray([
        Document(Prompt(prompt='dummystart')),
        Document(Prompt(prompt='dummyend'))
    ])
    identity: Text = ModelMetaIdentity

    def to_table(self):
        print('=' * 100)
        table = []
        unseen = ['identity']
        for attr, value in self.__dict__.items():
            if attr in unseen:
                continue
            if isinstance(value, list):
                value = ', '.join(str(item) for item in value)
            table.append([attr, value])
        max_attr_length = max(len(attr) for attr, _ in table)
        for attr, value in table:
            print(f'{attr.ljust(max_attr_length)} | {value}')


@dataclass
class ModelzooMeta:
    """
    A class representing the metadata of a model stored in a model zoo.

    Attributes:
    -----------
    modelzoo_name : str, optional
        The name of the model zoo. Defaults to the author's name.
    author_name : str, optional
        The author's name. Defaults to None.
    cache_dir : str, optional
        The local cache directory to store downloaded files.
        Defaults to '~/.cache/modelzoo'.
    oss_config_file: str, optional
        The file path to the configuration file for accessing OSS.

    Methods:
    --------
    __post_init__(self)
        A method to initialize the object after it has been instantiated.
    """

    modelzoo_name: Optional[str] = None
    author_name: Optional[str] = ''
    cache_dir: str = os.path.join(os.path.expanduser('~'), '.cache/modelzoo')
    oss_config_file: Optional[str] = None

    def __post_init__(self):
        """
        A method to initialize the object after it has been instantiated.
        """

        # Get the author's name and the model zoo's name
        # self.author_name = ModelScopeConfig.get_user_info()[0]
        if self.modelzoo_name is None:
            self.modelzoo_name = self.author_name

        # Create a local cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Log the object's properties
        logging.info('ModelzooMeta with modelzoo_name %s and author_name %s',
                     self.modelzoo_name, self.author_name)
        logging.info('ModelzooMeta with cache_dir %s', self.cache_dir)

        return
