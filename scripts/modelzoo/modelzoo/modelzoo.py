import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import List, Optional

import Levenshtein
import requests
from docarray import Document, DocumentArray

from .insert import insert_history_records
from .metainfo import ModelMeta, ModelzooMeta
from .utils import DEFAULT_REGION_MODEL, split_prompt


class ModelZoo:
    """
    A class representing a model zoo.

    Attributes:
    -----------
    modelzoo_name : str, optional
        The name of the model zoo. Defaults to the author's name.
    modelzoo_file : str, optional
        The file path to the model zoo binary file. Defaults to None.

    Properties:
    -----------
    meta : ModelzooMeta
        An instance of ModelzooMeta that contains metadata about the model zoo.
    modelzoo : DocumentArray
        A DocumentArray containing the models in the model zoo.

    Methods:
    --------
    __init__(self, modelzoo_name=None, modelzoo_file=None)
        Initializes the ModelZoo object.
    """
    def __init__(self,
                 modelzoo_name: Optional[str] = None,
                 modelzoo_file: Optional[str] = None,
                 region: Optional[str] = 'cn-hangzhou',
                 modelzoo_dir: Optional[str] = ''):
        """
        Initializes the ModelZoo object.
        """

        self.meta = ModelzooMeta(modelzoo_name=modelzoo_name,
                                 cache_dir=modelzoo_dir)

        if modelzoo_file is None:
            self.modelzoo_file = os.path.join(self.meta.cache_dir,
                                              'modelzoo.bin')
        else:
            self.modelzoo_file = modelzoo_file

        if os.path.exists(self.modelzoo_file):
            self.modelzoo = DocumentArray.load(self.modelzoo_file)
        else:
            self.modelzoo = DocumentArray()

    # ------------------------------ core API ------------------------------

    def save(self):
        # check if directory exists, create directory if not
        directory = os.path.dirname(self.modelzoo_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.modelzoo.save(self.modelzoo_file)

    def download_model(self,
                       model_name: str,
                       output_dir: str = None,
                       is_lora: bool = False,
                       region: str = 'cn-hangzhou') -> bool:
        """
        Downloads the specified model from ModelScope.

        Args:
            model_name (str): The name of the model to download.
            output_dir (str, optional): The output directory to
                save the downloaded model.

        Returns:
            None.

        Raises:
            ValueError: If the specified model does not exist
                in the local model zoo.
        """
        # Check if the model exists in the local model zoo and ModelScope
        ext = '.safetensors'
        model_name_ext = model_name + ext
        model_exists = self.check_model_exists(model_name_ext)
        if not model_exists:
            model_name_ext = model_name + '.ckpt'
            model_exists = self.check_model_exists(model_name_ext)
        if model_exists:
            # Set the output directory if it is not provided
            if output_dir is None:
                output_dir = self.meta.cache_dir
                logging.info(
                    'Without specifying output_dir, ' +
                    f'the downloaded model will be saved to: {output_dir}')

            # Download model using OSS
            mid_path = 'Stable-diffusion/'
            if is_lora:
                mid_path = 'Lora/'
            url = DEFAULT_REGION_MODEL.format(
                region) + mid_path + model_name_ext
            down_res = requests.get(url)
            if down_res.status_code == 404:
                return False
            with open(os.path.join(output_dir, model_name_ext), 'wb') as f:
                f.write(down_res.content)

        return True

    def create_model(self,
                     model_dir: str = '',
                     model_name: str = None,
                     model_description: str = '',
                     model_tags: List[str] = [],
                     before=True,
                     after=True,
                     oss_exists=False) -> Document:
        """
        Create_model function:
        1. Check if the model_name is empty,
            if it is, uses the directory in the model_dir as the model_name.
        2. Validate the provided model_name and generates new one if not valid.
        5. Create a ModelMeta object and stores it in the self.modelzoo.

        Args:
            model_dir (str): The path to the directory
                containing the model files to be uploaded.
            model_name (str, optional):
                The name of the model to be created.
                If not provided, the directory name in model_dir will be used.
            model_description (str, optional): A description of the model.
            model_tags (List[str], optional):
                A list of tags associated with the model. Defaults to [].

        Returns:
            None

        Raises:
            FileNotFoundError: If the model directory is not found.
            ValueError: If the model name already exists
                in the ModelZoo or is not valid.
            RuntimeError: If fails to create/upload the model.
        """

        # Check if model_name is empty
        if model_name is None:
            # Use the last directory in model_dir as the default model_name
            model_name = os.path.basename(model_dir)

        model_name_no_ext = os.path.splitext(model_name)[0]
        if len(model_tags) == 0:
            model_tags = split_prompt(model_description, ',')
            logging.info(
                f'with no model_tags input : we use {model_description}' +
                f' to generate some tags {model_tags}')

        # Validate model_name using a custom function
        if not self.is_valid_model_name(model_name_no_ext):
            # Generate a new model_name using model_description if not valid
            new_model_name = self.generate_valid_model_name(
                model_description, model_tags)
            raise ValueError(
                f'The provided model name {model_name_no_ext} is not valid.' +
                f' We generated a new name: {new_model_name}. ≠Please run ' +
                'create_model again with the generated name.')

        model_file_list = [model_name]
        model_tags.append('author_name_' + self.meta.author_name)
        model_tags.append('modelzoo_name_' + self.meta.modelzoo_name)

        # Create ModelMeta and store it in the self.modelzoo database
        model_meta = ModelMeta(
            model_name=model_name_no_ext,
            model_description=model_description,
            model_version='initial',
            model_tags=model_tags,
            model_created=datetime.now().strftime('%Y%m%d-%H%M%S'),
            model_updated=datetime.now().strftime('%Y%m%d-%H%M%S'),
            model_file_list=model_file_list)

        self.modelzoo.append(Document(model_meta))
        self.save()

        return self.modelzoo[-1]

    def delete_model(self, model_name: str) -> str:
        """
        Deletes a model with the specified name from the model zoo.

        Args:
            model_name (str): The name of the model to delete.

        Returns:
            str: A message whether the model was successfully deleted.

        Raises:
            ValueError: If the specified model does not exist in the model zoo.
        """
        # Check if the model exists
        model = self.get_model_by_name(model_name)
        if model is None:
            logging.info(f"The model '{model_name}' does not exist.")

        # Delete the model from the model zoo
        del self.modelzoo[model.id]
        self.save()

        # Return a success message
        return f"The model '{model_name}' has been successfully deleted."

    # -------------------------- util function  --------------------------

    def refresh_view(self):
        self.modelzoo = DocumentArray(
            list(self.modelzoo),
            config={'n_dim': 256},
            subindex_configs={
                '@.[model_name]': {
                    'n_dim': 256
                },
                '@.[model_tags]': {
                    'n_dim': 256
                },
                '@.[history_records].[prompt]': {
                    'n_dim': 256
                },
                '@.[history_records].[negative_prompt]': {
                    'n_dim': 256
                },
            })

    def is_valid_model_name(self, model_name):
        # this logits will supported when modelscope is ready
        if 0:
            model_exists_modelscope = False
            try:
                self.modelscope.get_model('/'.join(
                    [self.meta.modelzoo_name, model_name]))
                model_exists_modelscope = True
            except requests.exceptions.HTTPError as error:
                if error.response.status_code == 404:
                    model_exists_modelscope = False
                else:
                    raise ValueError(
                        'check model_name with modelscope:  ' +
                        'http://www.modelscope.cn/api/v1/models/ ' +
                        'encounter network error')
            if model_exists_modelscope:
                raise ValueError(
                    'The provided model name {model_name} is not ' +
                    f'already in modelscope/{self.meta.modelzoo_name} ' +
                    'Please run create_model again with the another name. ')

        return self.get_model_by_name(model_name) is None

    def generate_valid_model_name(self, model_description, model_tags):
        first_2_tag = model_tags[:2]
        first_2_tag += [self.meta.author_name, self.meta.modelzoo_name]
        return '_'.join(first_2_tag)

    def check_model_exists(self, model_name: str) -> bool:
        """
        Checks if a model with the specified name exists in
            both the local model zoo and ModelScope.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model exists in both
                the local model zoo and ModelScope, False otherwise.

        Raises:
            ValueError: If the specified model does not
                exist in the local model zoo.
        """
        # Check if the model exists in the local model zoo
        orig_name = os.path.splitext(model_name)[0]
        model = self.get_model_by_name(orig_name)
        if model is None:
            raise ValueError(f"The model '{orig_name}' does " +
                             'not exist in the local model zoo.')

        exists = self.bucket.object_exists(
            os.path.join(self.user_dir, model_name))
        return exists

    # ---------------------------- modelzoo API ----------------------------

    def list_model(self) -> List[str]:
        """
        List all model names

        :return: List of all model names
        """
        return self.modelzoo['@.[model_name]'].texts

    def get_model_by_name(self, name: str, **kwargs) -> Optional[Document]:
        """
        Get model information by name

        :param name: Name of the model
        :return: Matching model information or None if not found
        """
        case_insensitive = kwargs.pop('case_insensitive', False)
        pattern = '$eq'
        if case_insensitive:
            pattern = '$regex'
        match = self.modelzoo['@.[model_name]'].find({'text': {pattern: name}})
        if len(match) > 0:
            return self.modelzoo[match[0].parent_id]
        else:
            return None

    def get_model_by_tag(self, tag: str, **kwargs) -> Optional[List[Document]]:
        """
        Get model information by tag

        :param tag: Tag of the model
        :return: List of matching model information or None if not found
        """
        self.modelzoo = DocumentArray(self.modelzoo)

        case_insensitive = kwargs.pop('case_insensitive', False)
        pattern = '$eq'
        if case_insensitive:
            pattern = '$regex'
        match = self.modelzoo['@.[model_tags]'].find(
            {'text': {
                pattern: '(?i)' + tag
            }})

        return_id_set = []
        for k in match:
            return_id_set.append(self.modelzoo['@c'][k.parent_id].parent_id)
        return_id_set = list(OrderedDict.fromkeys(return_id_set))

        if len(return_id_set) > 0:
            return [self.modelzoo[rid] for rid in return_id_set]
        else:
            return None

    # TODO: update to embedding
    def search_model_by_name(self, name: str,
                             **kwargs) -> Optional[List[Document]]:
        """
        Search models by name

        :param name: Name of the model
        :param topk: Number of top results to return. Default is 3.
        :return: List of matching model information or None if not found
        """
        self.refresh_view()

        case_insensitive = kwargs.pop('case_insensitive', False)

        def func(x):
            if case_insensitive:
                return x.lower()
            return x

        return_id_set = sorted(self.modelzoo,
                               key=lambda doc: Levenshtein.ratio(
                                   func(name), func(doc.model_name.text)),
                               reverse=True)
        topk = kwargs.pop('topk', 3)
        if topk != -1:
            return_id_set = return_id_set[:min(len(return_id_set), topk)]
        if len(return_id_set) > 0:
            return return_id_set
        else:
            return None

    # TODO: update to embedding
    def search_model_by_tag(self, tag: str,
                            **kwargs) -> Optional[List[Document]]:
        """
        Search models by tag

        :param tag: Tag of the model
        :param topk: Number of top results to return. Default is 3.
        :return: List of matching model information or None if not found
        """
        self.refresh_view()

        case_insensitive = kwargs.pop('case_insensitive', False)

        def func(x):
            if case_insensitive:
                return x.lower()
            return x

        r = list()
        for m in self.modelzoo:
            r.extend([{'tag': t.text, 'id': m.id} for t in m.model_tags])
        match = sorted(
            r,
            key=lambda doc: Levenshtein.ratio(func(tag), func(doc['tag'])),
            reverse=True)
        return_id_set = []
        topk = kwargs.pop('topk', 3)

        def key_func(x):
            return x['id']

        key_list = [key_func(x) for x in match]
        unique_keys = OrderedDict.fromkeys(key_list).keys()
        return_id_set = [self.modelzoo[k] for k in unique_keys]
        if topk != -1:
            return_id_set = return_id_set[:min(len(return_id_set), topk)]
        if len(return_id_set) > 0:
            return return_id_set
        else:
            return None

    def search_history_records_by_keyword(
            self, model_name, keyword: str,
            **kwargs) -> Optional[List[Document]]:
        """
        :return: List of matching prompt information or None if not found.
        """
        self.refresh_view()

        matched_prompt = list()
        model = self.get_model_by_name(model_name)
        if model is not None:
            matched_prompt = model.history_records[
                '@.[prompt],.[negative_prompt]'].find(
                    {'text': {
                        '$regex': keyword
                    }})

        return_id_set = []
        for k in matched_prompt:
            return_id_set.append(k.parent_id)
        return_id_set = list(OrderedDict.fromkeys(return_id_set))

        if len(return_id_set) > 0:
            return [model.history_records[rid] for rid in return_id_set]
        else:
            return None

    def delete_history_record_by_id(self, model_name, prompt_id):
        """

        """
        prompts = self.get_model_by_name(model_name).history_records
        del prompts[prompt_id]
        self.save()

    def __call__(self):
        return self.modelzoo

    def insert_image(self, model_name, prompt, image_path, **kwargs):
        prompt.output_image = image_path
        meta = self.get_model_by_name(model_name)

        insert_history_records(meta, prompt)

    def get_filename_by_modelname(self, model_name):
        ext = '.safetensors'
        ext_model = model_name + ext
        if self.bucket.object_exists(ext_model):
            return ext_model
        else:
            return model_name + '.ckpt'
