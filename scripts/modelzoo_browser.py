import csv
import datetime
import hashlib
import json
import logging
import os
import platform
import re
import sys
import shutil
import tempfile
import traceback
from io import BytesIO, StringIO
from itertools import chain

import gradio as gr
import requests
import urllib3
from PIL import Image, UnidentifiedImageError

import modules.extras
import modules.images
import modules.ui
from modules import images, paths, script_callbacks, sd_models, shared
from modules.shared import cmd_opts, opts
from modules.ui_common import plaintext_to_html
from modules.ui_components import DropdownMulti, ToolButton
from scripts.modelzoo.modelzoo.metainfo import ModelMeta
from scripts.modelzoo.modelzoo.modelzoo import ModelZoo
from scripts.modelzoo.modelzoo.prompt import Prompt

import time
from modules import extensions
import html

default_tab_options = ['Gallery', 'Upload Model', 'Download Model']
tabs_list = [
    tab.strip() for tab in chain.from_iterable(
        csv.reader(StringIO(opts.modelzoo_browser_active_tabs))) if tab
] if hasattr(opts, 'modelzoo_browser_active_tabs') else default_tab_options

components_list = [
    'Sort by', 'keyword search', 'Tag keyword search', 'Generation Info',
    'File Name', 'File Time', 'Change Model', 'Send to buttons',
    'Copy to directory', 'Gallery Controls Bar', 'Ranking Bar', 'Delete Bar',
    'Additional Generation Info'
]

up_symbol = '\U000025b2'  # ▲
down_symbol = '\U000025bc'  # ▼
copy_move = ['Move', 'Copy']
copied_moved = ['Moved', 'Copied']
is_prompt = False  # Distinguish between the model and prompt mode.
unsaved_list = list(
)  # Stores models that have deteced not yet been uploaded to modelzoo.
lora_subdir = 'Lora'
model_dir = 'Stable-diffusion'
controlnet_subdir = 'ControlNet'
stable_diffusion_dir = os.path.abspath(
    os.path.join(paths.models_path, model_dir))
lora_dir = os.path.abspath(os.path.join(paths.models_path, lora_subdir))
controlnet_dir = os.path.abspath(
    os.path.join(paths.models_path, controlnet_subdir))
modelzoo_script_path = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))
default_image = os.path.join(modelzoo_script_path, 'scripts/default_image.jpg')
region = os.environ.get('SERVICE_NAME',
                        'sdwebui_zf@cn-zhangjiakou').split('@')[-1]
DEFAULT_MODEL_PREFIX = 'default_'
modelzoo_dir = os.path.abspath(os.path.join(paths.models_path, os.pardir))

saved_images = list()  # cache images that have been stored in modelzoo today.
path_maps = {
    'txt2img': opts.outdir_samples or opts.outdir_txt2img_samples,
    'img2img': opts.outdir_samples or opts.outdir_img2img_samples,
    'txt2img-grids': opts.outdir_grids or opts.outdir_txt2img_grids,
    'img2img-grids': opts.outdir_grids or opts.outdir_img2img_grids,
    'Extras': opts.outdir_samples or opts.outdir_extras_samples,
}
mz = ModelZoo(modelzoo_name='modelzoo_browser',
              region=region,
              modelzoo_dir=modelzoo_dir)
# By default, all images already exist in modelzoo at startup
start_day = datetime.date.today().strftime('%Y-%m-%d')
today_path_txt = os.path.join(path_maps['txt2img'], start_day)
today_path_img = os.path.join(path_maps['img2img'], start_day)
if os.path.exists(today_path_txt):
    for image in os.listdir(today_path_txt):
        image_path = os.path.join(today_path_txt, image)
        saved_images.append(image_path)
if os.path.exists(today_path_img):
    for image in os.listdir(today_path_img):
        image_path = os.path.join(today_path_img, image)
        saved_images.append(image_path)
source_model_dir = ""
target_model_dir = ""
user_data_dir = paths.models_path
# try:
#     public_data_dir = os.path.join(cmd_opts.shared_dir, 'models')
# except:
#     public_data_dir = user_data_dir
public_data_dir = user_data_dir
if cmd_opts.uid is not None:
    public_data_dir = os.path.join(os.path.abspath(os.path.dirname(user_data_dir) + os.path.sep, "."), "models") 
print("=" * 100)
print("cmd_opts.uid: ", cmd_opts.uid)
print("public_data_dir: ", public_data_dir)
print("=" * 100)
unmodel_list = ('png', 'md', 'info')
public_cache_dir = '/stable-diffusion-cache/models'
download_for_public = ('annotator', 'clip', 'Codeformer', 'ControlNet', 'SwinIR')
show_public_models = list()


class ModelZooBrowserTab():

    seen_base_tags = set()

    def __init__(self, name: str):
        self.name: str = os.path.basename(name) if os.path.isdir(
            name) else name
        self.path: str = os.path.realpath(path_maps.get(name, name))
        removed_tag = self.remove_invalid_html_tag_chars(self.name).lower()
        self.base_tag: str = 'modelzoo_browser_tab_' + \
            f'{self.get_unique_base_tag(removed_tag)}'

    def remove_invalid_html_tag_chars(self, tag: str) -> str:
        # Removes any character that is not a letter,
        # a digit, a hyphen, or an underscore
        removed = re.sub(r'[^a-zA-Z0-9\-_]', '', tag)
        return removed

    def get_unique_base_tag(self, base_tag: str) -> str:
        counter = 1
        while base_tag in self.seen_base_tags:
            match = re.search(r'_(\d+)$', base_tag)
            if match:
                counter = int(match.group(1)) + 1
                base_tag = re.sub(r'_(\d+)$', f'_{counter}', base_tag)
            else:
                base_tag = f'{base_tag}_{counter}'
            counter += 1
        self.seen_base_tags.add(base_tag)
        return base_tag

    def __str__(self):
        return f'Name: {self.name} / Path: {self.path} / ' + \
            f'Base tag: {self.base_tag} / ' + \
            f'Seen base tags: {self.seen_base_tags}'


tabs_list = [ModelZooBrowserTab(tab) for tab in tabs_list]

# Logging
logger = logging.getLogger(__name__)
logger_mode = logging.ERROR
if hasattr(opts, 'modelzoo_browser_logger_warning'):
    if opts.modelzoo_browser_logger_warning:
        logger_mode = logging.WARNING
if hasattr(opts, 'modelzoo_browser_logger_debug'):
    if opts.modelzoo_browser_logger_debug:
        logger_mode = logging.DEBUG
logger.setLevel(logger_mode)
if (logger.hasHandlers()):
    logger.handlers.clear()
console_handler = logging.StreamHandler()
console_handler.setLevel(logger_mode)
logger.addHandler(console_handler)
# Debug logging
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f'{sys.executable} {sys.version}')
    logger.debug(f'{platform.system()} {platform.version()}')
    try:
        git = os.environ.get('GIT', 'git')
        commit_hash = os.popen(f'{git} rev-parse HEAD').read()
    except Exception as e:
        commit_hash = e
    logger.debug(f'{commit_hash}')
    logger.debug(f'Gradio {gr.__version__}')
    logger.debug(f'{paths.script_path}')
    with open(cmd_opts.ui_config_file, 'r') as f:
        logger.debug(f.read())
    with open(cmd_opts.ui_settings_file, 'r') as f:
        logger.debug(f.read())
    logger.debug(os.path.realpath(__file__))
    logger.debug([str(tab) for tab in tabs_list])

def convert_size(file_size):
    """
    """
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = 1024
    for i in range(len(units)):
        if (file_size / size) < 1:
            return "%.2f%s" % (file_size, units[i])
        file_size = file_size / size


def convert_model_type(model_type):
    if model_type == 'Lora':
        return 'lora'
    if model_type == 'Stable-diffusion':
        return 'safetensors'
    
    return None


def correlated_info(model_name_with_suffix):
    image_suffix = ".preview.png"
    info_suffix = ".civitai.info"
    model_name = os.path.splitext(model_name_with_suffix)[0]

    return (model_name+image_suffix, model_name+info_suffix)


def sort_order_flip(turn_page_switch: int, sort_order: str):
    """
    This function flips the sort order.

    Args:
        turn_page_switch: A boolean value indicating whether to turn the page.
        sort_order: The current sort order.

    Returns:
        A tuple of the following:
        1: A value to indicate that the sort order was flipped.
        -turn_page_switch: The value of the turn page switch, negated.
        sort_order: The new sort order.
    """

    # Check the current sort order.
    if sort_order == up_symbol:
        # If the current sort order is up, set the new sort order to down.
        sort_order = down_symbol
    else:
        # If the current sort order is down, set the new sort order to up.
        sort_order = up_symbol
    return 1, -turn_page_switch, sort_order


def delete_modelzoo_item(delete_num: int, name: str, filenames: list,
                         image_index: int, visible_num: int,
                         turn_page_switch: int, select_image_switch: int,
                         image_page_list: list, prompt_ids: list):
    """
    This function deletes the image with the given index.
    is_prompt=True, the corresponding selected image is deleted
    is_prompt=False, the corresponding selected model is deleted

    Args:
        tab_base_tag_box: The HTML tag for the base of the tab.
        delete_num: The index of the image to delete.
        name: The name of the model that generated the image.
        filenames: A list of the filenames of the images.
        image_index: The index of the image in the list of filenames.
        visible_num: The number of images that are visible on the page.
        turn_page_switch: A boolean value indicating whether to turn the page.
        select_image_switch: A boolean value whether to select the image.
        image_page_list: A list of tuples of images and their names.
        prompt_ids: A list of the prompt IDs for the images.

    Returns:
        A tuple of the following:
        new_file_list: A list of the filenames of the images.
        new_image_page_list: A list of tuples of images and their names.
        load_info: A message to display at the top of the page.
        prompt_ids: A list of the prompt IDs for the images.
    """

    global mz, is_prompt
    refresh = False
    delete_num = int(delete_num)
    image_index = int(image_index)
    visible_num = int(visible_num)
    if image_page_list != '':
        image_page_list = json.loads(image_page_list)
    else:
        refresh = True
    new_file_list = []
    new_image_page_list = list()

    if name == '':
        refresh = True
    else:
        try:
            index_files = filenames.index(name)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            # Something went wrong, force a page refresh
            refresh = True

    # If something goes wrong, a refresh is forced and no deletion is performed
    if refresh:
        turn_page_switch = -turn_page_switch
    # If the deletion process is smooth, select the next picture after deletion
    else:
        select_image_switch = -select_image_switch

    if not refresh:
        if is_prompt:
            new_file_list = filenames
            try:
                mz.delete_history_record_by_id(name, prompt_ids[image_index])
            except Exception:
                refresh = True
            new_image_page_list = image_page_list[:image_index] + \
                image_page_list[image_index + 1:]
            prompt_ids = prompt_ids[:image_index] + prompt_ids[image_index +
                                                               1:]
            load_info = "<div style='color:#999' align='center'>"
            load_info += f'{len(new_image_page_list)} ' + \
                'matched prompts in {name}'
            load_info += '</div>'
            if image_index + 1 >= len(image_page_list):
                select_image_switch = -select_image_switch
            return new_file_list, 1, turn_page_switch, visible_num, \
                new_image_page_list, select_image_switch, \
                json.dumps(new_image_page_list), load_info, prompt_ids
        else:
            new_file_list = filenames[:index_files] + \
                filenames[index_files+1:]
            mz.delete_model(name)
            new_image_page_list = image_page_list[:image_index] + \
                image_page_list[image_index + 1:]
            if image_index + 1 >= len(image_page_list):
                select_image_switch = -select_image_switch

    load_info = "<div style='color:#999' align='center'>"
    load_info += f'{len(new_image_page_list)} matched models in modelzoo'
    load_info += '</div>'

    return new_file_list, 1, turn_page_switch, visible_num, \
        new_image_page_list, select_image_switch, \
        json.dumps(new_image_page_list), load_info, prompt_ids


def get_page_by_keyword(filenames: list, name_keyword: str, sort_by: str,
                        sort_order: str, search_model_type_select: str,
                        case_insensitive: bool, topk: int,
                        accurate_match: bool, model_name: str):
    """
    This function gets the page of images for the given parameters.
    is_prompt=True, images (prompt/negative_prompt contains name_keyword)
        corresponding to model_name is returned.
    is_prompt=False, returns a list of models sorted by search criteria.

    0. detect all today's images and save which has not been saved.
    1. search images/models by keyword.
    2. sort images/models by type of sort_by.

    Args:
        filenames: A list of filenames of the images on the page.
        name_keyword: A keyword to search for in the image names.
        sort_by: The field to sort the images by.
        sort_order: The order to sort the images in.
        tab_base_tag_box: The tag box to search for images in.
        search_model_type_select: The type of images to search for.
        case_insensitive: Whether to ignore case when searching for names.
        topk: The number of images to return.
        accurate_match: Whether to require an exact match for the name keyword.
        model_name: The name of the model to search for images for.

    Returns:
        A tuple of the following:
        filenames: A list of filenames of the images on the page.
        images: A list of images on the page.
        models: A list of models that the images belong to.
        model_names: A list of names of the models that the images belong to.
        load_info: A message to display at the top of the page.
    """

    global start_day, today_path_txt, today_path_img, \
        saved_images, mz, is_prompt

    today = datetime.date.today().strftime('%Y-%m-%d')
    if today != start_day:
        start_day = today
        saved_images = []
        today_path_txt = os.path.join(path_maps['txt2img'], today)
        today_path_img = os.path.join(path_maps['img2img'], today)

    today_path = today_path_txt
    if os.path.exists(today_path):
        for image in os.listdir(today_path):
            image_path = os.path.join(today_path, image)
            if image_path in saved_images:
                continue
            content = Image.open(image_path)
            _, prompt = Prompt.convertFromImage(content)
            lora_names = check_lora(prompt.prompt)
            lora_names.append(prompt.model)
            for name in lora_names:
                model = mz.get_model_by_name(name)
                if model is not None:
                    mz.insert_image(name, prompt=prompt, image_path=image_path)
            saved_images.append(image_path)
    today_path = today_path_img
    if os.path.exists(today_path):
        for image in os.listdir(today_path):
            image_path = os.path.join(today_path, image)
            if image_path in saved_images:
                continue
            content = Image.open(image_path)
            _, prompt = Prompt.convertFromImage(content)
            lora_names = check_lora(prompt.prompt)
            lora_names.append(prompt.model)
            for name in lora_names:
                model = mz.get_model_by_name(name)
                if model is not None:
                    mz.insert_image(name, prompt=prompt, image_path=image_path)
            saved_images.append(image_path)
    mz.save()

    try:
        topk = (int)(topk)
    except ValueError:
        logger.warning(f'{topk} is not a resonable int value.')
        topk = 10

    # Set temp_dir from webui settings, so gradio uses it
    if shared.opts.temp_dir != '':
        tempfile.tempdir = shared.opts.temp_dir

    return_image_list = list()
    flag = False
    models = list()
    image_list = list()
    prompt_ids = list()

    if is_prompt:
        models = mz.search_history_records_by_keyword(model_name, name_keyword)
        if models is not None:
            for model in models:
                current_model = Prompt(model)
                if current_model.output_image is None:
                    continue
                image = current_model.output_image
                return_image_list.append((image, ''))
                prompt_ids.append(model.id)

        load_info = "<div style='color:#999' align='center'>"
        load_info += f'{len(return_image_list)} matched ' + \
            'prompts in {model_name}'
        load_info += '</div>'

        return filenames, return_image_list, model_name, '', 0, \
            load_info, json.dumps(return_image_list), \
            gr.update(value='back to model zoo'), prompt_ids

    filenames = list()
    models = mz.list_model()
    if search_model_type_select == 'checkpoints':
        search_model_type_select = 'safetensors'
    if len(name_keyword) != 0 and search_model_type_select == 'all':
        if accurate_match:
            models = [mz.get_model_by_name(name_keyword)]
        else:
            models = mz.search_model_by_name(name_keyword,
                                             topk=topk,
                                             case_insensitive=case_insensitive)
        flag = True
    elif search_model_type_select != 'all' and len(name_keyword) == 0:
        models = mz.get_model_by_tag(search_model_type_select,
                                     case_insensitive=case_insensitive)
        flag = True
    elif len(name_keyword) != 0 and search_model_type_select != 'all':
        key_models = mz.search_model_by_name(name_keyword,
                                             topk=-1,
                                             case_insensitive=case_insensitive)
        tag_models = mz.get_model_by_tag(search_model_type_select,
                                         topk=-1,
                                         case_insensitive=case_insensitive)
        models = modelmeta_intersection(tag_models, key_models)
        flag = True
    if not flag:
        if models is not None:
            models = [mz.get_model_by_name(m) for m in models]
    if topk != -1:
        models = models[:topk]

    reverse = True
    if sort_order == up_symbol:
        reverse = False

    if models is not None and models != [] and models != [None]:
        if sort_by == 'updated date':
            models = sorted(models,
                            key=lambda x: x.model_updated.text,
                            reverse=reverse)
        elif sort_by == 'model name':
            models = sorted(models,
                            key=lambda x: x.model_name.text,
                            reverse=reverse)
        elif sort_by == 'created date':
            models = sorted(models,
                            key=lambda x: x.model_created.text,
                            reverse=reverse)
        elif sort_by == 'version':
            models = sorted(models,
                            key=lambda x: x.model_version.text,
                            reverse=reverse)
        elif sort_by == 'author':
            models = sorted(models,
                            key=lambda x: x.model_author.text,
                            reverse=reverse)
        elif sort_by == 'dataset':
            models = sorted(models,
                            key=lambda x: x.model_dataset.text,
                            reverse=reverse)
        elif sort_by == 'sampler':
            models = sorted(models,
                            key=lambda x: x.model_default_sampler.text,
                            reverse=reverse)
        elif sort_by == 'cfg scale':
            models = sorted(models,
                            key=lambda x: x.model_default_cfg_scale.text,
                            reverse=reverse)
        for model in models:
            current_model = ModelMeta(model)
            image = default_image
            if len(current_model.history_records) > 2:
                for p in reversed(current_model.history_records):
                    if p.output_image is not None:
                        image = p.output_image
                        break
            return_image_list.append((image, current_model.model_name))
            filenames.append(current_model.model_name)
            image_list.append(image)

    load_info = "<div style='color:#999' align='center'>"
    load_info += f'{len(return_image_list)} matched models in modelzoo'
    load_info += '</div>'

    is_prompt = False

    return filenames, return_image_list, model_name, '', \
        0, load_info, json.dumps(image_list), \
        gr.update(value='show prompt zoo'), prompt_ids


def view_modelzoo_item_info(tab_base_tag: str, num: int, filenames: list,
                            turn_page_switch: int, prompt_ids: list,
                            model_name: str):
    """
    This function shows the information for the given item.
    When is_prompt=True, the function will get the prompt, negative_prompt
        and other information from modelzoo based on the passed image.
    When is_prompt=False, this function returns the model_name of the model.

    Args:
        tab_base_tag_box: The HTML tag for the base of the tab.
        num: The index of the image.
        page_index: The index of the page.
        filenames: A list of the filenames of the images.
        turn_page_switch: A boolean value indicating whether to turn the page.
        prompt_ids: A list of the prompt IDs for the images.
        model_name: The name of the model that generated the images.

    Returns:
        A tuple of the following:
        filename: The filename of the image.
        tm: The time stamp for the image.
        num: The index of the image.
        file: The image file.
        turn_page_switch: The value of the turn page switch.
        info: A message to display at the top of the page.
        image_pil: A PIL image of the image.
        pnginfo: A dictionary of information about the image.
        prompt: The prompt for the image.
        negative_prompt: The negative prompt for the image.
        prompt_sampler: The prompt sampler for the image.
        prompt_seed: The prompt seed for the image.
        prompt_cfg_scale: The prompt cfg scale for the image.
        prompt_steps: The number of steps for the image.
        gr_json: A JSON string of the image information.
    """
    global is_prompt
    num = (int)(num)
    if not is_prompt:
        file = filenames[num]

        return filenames[num], num, file, turn_page_switch, None, \
            '', '', '', '', '', '', '', gr.update(visible=True)
    elif is_prompt:
        prompt_id = prompt_ids[num]
        model = mz.get_model_by_name(model_name)
        prompt_sampler = ''
        prompt_seed = ''
        prompt_cfg_scale = ''
        prompt_steps = ''
        prompt = ''
        negative_prompt = ''
        output_image = default_image
        image_pil = None
        if model is not None:
            p = Prompt(model.history_records[prompt_id])
            prompt = p.prompt
            negative_prompt = p.negative_prompt
            prompt_sampler = p.sampler
            prompt_seed = p.seed
            prompt_steps = str(p.steps)
            prompt_cfg_scale = str(p.cfg_scale)
            output_image = p.output_image
            if output_image.startswith('http'):
                res = requests.get(output_image)
                image_pil = Image.open(BytesIO(res.content))
            else:
                image_pil = Image.open(output_image)

        return model_name, num, output_image, turn_page_switch, \
            image_pil, run_pnginfo(image_pil), prompt, negative_prompt, \
            prompt_sampler, prompt_seed, prompt_cfg_scale, prompt_steps, \
            gr.update(visible=True)


def upload_model(is_lora: bool, filename: str = ''):
    """
    This function uploads the given model to the model zoo.

    Args:
        model_file: The file containing the model to upload.
        is_lora: A boolean value indicating whether the model is a lora model.
        filename: The name of the model file.
            If not specified, the name of the file will be used.

    Returns:
        A message to display at the top of the page.
    """
    global mz

    # Set the output directory.
    model_dir = stable_diffusion_dir
    model_tags = ['safetensors']
    if is_lora:
        model_dir = lora_dir
        model_tags = ['lora']

    # Get the name of the model.
    model_name = os.path.splitext(filename)[0]
    target_model = os.path.join(model_dir, filename)
    if mz.get_model_by_name(model_name) is not None:
        return '<div>model already exists in modelzoo</div>'
    try:
        # Upload to disk
        # if model_file is not None:
        #     os.replace(model_file.name, target_model)
        # Upload to ModelZoo
        mz.create_model(target_model, filename, model_tags=model_tags)
    except Exception:
        # Upload to disk
        # if model_file is not None:
        #     os.replace(model_file.name, target_model)
        # Upload to ModelZoo
        mz.create_model(target_model, filename, model_tags=model_tags)
    # Create a message to display at the top of the page.
    load_info = "<div style='color:#111' align='center'>"
    load_info += 'Upload Model success'
    load_info += '</div>'

    return load_info


def load_model(model_name: str, select_hidden: int):
    """
    This function downloads the given model to the local machine.

    Args:
        model_name: The name of the model to download.
        select_hidden: A boolean value indicating
            whether to select the hidden layers of the model.

    Returns:
        A tuple of the following:
        load_info: A message to display at the top of the page.
        select_hidden: The value of the select_hidden argument.
            if change, the selected model will correspondly change.
        model_name: The name of the model that was downloaded.
    """

    # Check if the model is already downloaded.
    skip_download = False
    if model_name in [
            model.model_name for model in sd_models.checkpoints_list.values()
    ]:
        skip_download = True
    # Check if the model is already in the lora directory.
    if not skip_download:
        for model in os.listdir(lora_dir):
            if model_name == os.path.splitext(model)[0]:
                skip_download = True

    # Get the model from the model zoo.
    model = mz.get_model_by_name(model_name)

    # Set the output directory.
    output_dir = stable_diffusion_dir
    is_lora = False
    if model is not None:
        # Check if the model is a lora model.
        if 'lora' in [x.lower() for x in model.model_tags.texts]:
            output_dir = lora_dir
            is_lora = True

    # Create a message to display at the top of the page.
    load_info = "<div style='color:#999' align='center'>"

    # If the model is a lora model, display a message and return.
    if is_lora:
        load_info += 'lora cannot be selected'
        load_info += '</div>'
        return load_info, select_hidden, model_name

    # If the model is not already downloaded, download it.
    if not skip_download and model_name.startswith(DEFAULT_MODEL_PREFIX):
        res = mz.download_model(model_name,
                                output_dir=output_dir,
                                is_lora=is_lora,
                                region=region)
        if res:
            load_info += 'download successfully'
            # Refresh the checkpoints list.
            shared.refresh_checkpoints()
        else:
            load_info += 'download failed'
            select_hidden = -select_hidden

    # If the model is not already downloaded and it is not a default model,
    # display a message.
    if not skip_download and not model_name.startswith(DEFAULT_MODEL_PREFIX):
        select_hidden = -select_hidden
        load_info += 'model not exists in local'

    # Add a closing tag to the message.
    load_info += '</div>'

    # Return the message, the value of select_hidden, and the name of the model
    return load_info, -select_hidden, model_name


def check_lora(prompt: str):
    """
    This function checks if the given prompt
    contains a lora model and its weight.

    Args:
        prompt: The prompt to check.

    Returns:
        A list of models that were found in the prompt.
    """

    # Create a list to store the models.
    return_list = list()

    # Create a regular expression pattern to match lora models.
    pattern = r'<lora:(.*?):(0\.\d*[1-9]|1|1\.\d*[1-9])>'

    # Find all matches for the pattern in the prompt.
    for model, _ in re.findall(pattern, prompt):
        # Add the model to the list.
        return_list.append(model)

    # Return the list of models.
    return return_list


def control_promptzoo(model_name: str, turn_page_switch: int,
                      prompt_ids: list):
    """
    This function gets the prompts for the given model.
    For is_prompt=True, the current click is back to modelzoo,
        leaving all prompt related content blank and updating some ui.
    If is_prompt=False, the current click is show prompt zoo,
        and search prompt corresponding to this model in modelzoo.bin
            according to model_name.

    Args:
        model_name: The name of the model.
        turn_page_switch: A boolean value indicating whether to turn the page.
        prompt_ids: A list of the prompt IDs.

    Returns:
        A tuple of the following:
        image_list: A list of images for the prompts.
        update_panel: A boolean value indicating whether to update the panel.
        update_button: The value of the update button.
        turn_page_switch: The value of the turn page switch.
        prompt_ids: A list of the prompt IDs.
        load_info: A message to display at the top of the page.
        gr_json: A JSON string of the image list. for some ui component change.
    """
    # Set the global variables.
    global is_prompt
    update_panel = gr.update(visible=False)
    # Create a button to show the prompt zoo.
    update_button = gr.update(value='show prompt zoo')

    # Check if the prompt mode is already enabled.
    if is_prompt:
        # Disable the prompt mode.
        is_prompt = False
        return [], update_panel, update_button, \
            -turn_page_switch, [], '', '', '', '', '', '', \
            None, '', '', gr.update(label='model search', value=''), \
            gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=True), gr.update(visible=False), \
            gr.update(visible=True)

    # Enable the prompt mode.
    is_prompt = True
    # Update the button to go back to the model zoo.
    update_button = gr.update(value='back to model zoo')
    # Get the model from the model zoo.
    model = mz.get_model_by_name(model_name)

    image_list = list()
    prompt_ids = list()
    for prompt in model.history_records:
        p = Prompt(prompt)
        output_image = p.output_image
        if output_image is not None:
            image_list.append((output_image, ''))
            prompt_ids.append(prompt.id)
    length = len(image_list)
    load_info = "<div style='color:#999' align='center'>"
    load_info += f'{length} matched prompts in {model_name}'
    load_info += '</div>'

    return image_list, gr.update(
        visible=True
    ), update_button, turn_page_switch, prompt_ids, \
        '', '', '', '', '', '', \
        None, load_info, json.dumps(image_list), \
        gr.update(label='prompt search', value=''), \
        gr.update(visible=True), gr.update(visible=True), \
        gr.update(visible=False), gr.update(visible=True), \
        gr.update(visible=False)


def upload_all_models():
    """
    This function uploads all models to the model zoo.

    Args:
        None.

    Returns:
        A tuple of the following:
        load_info: A message to display at the top of the page.
        gr_json: A JSON string of the list of models that were uploaded.
    """

    global mz, unsaved_list

    if unsaved_list != []:
        # save all models
        for unsaved_model in unsaved_list:
            model_name, is_lora = unsaved_model
            upload_model(is_lora, filename=model_name + '.safetensors')
        sd_unsaved = [model for model, is_lora in unsaved_list if not is_lora]
        lora_unsaved = [model for model, is_lora in unsaved_list if is_lora]
        load_info = "<div style='color:#999' align='center'>"
        load_info += f"SD: {', '.join(sd_unsaved)}" + \
            f' <br> total {len(sd_unsaved)} models saved in modelzoo <br>'
        load_info += f"LoRA: {', '.join(lora_unsaved)}" + \
            f' <br> total {len(lora_unsaved)} lora saved in modelzoo'
        unsaved_list = []
        load_info += '</div>'
        return load_info, gr.update(value='detect all exists models',
                                    variant='secondary')
    for model in sd_models.checkpoints_list.values():
        model_name = model.model_name
        if mz.get_model_by_name(model_name) is None:
            unsaved_list.append((model_name, False))
    for model in os.listdir(lora_dir):
        model_name = os.path.splitext(model)[0]
        if mz.get_model_by_name(model_name) is None:
            unsaved_list.append((model_name, True))

    load_info = "<div style='color:#111' align='center'>"
    sd_unsaved = [model for model, is_lora in unsaved_list if not is_lora]
    lora_unsaved = [model for model, is_lora in unsaved_list if is_lora]
    load_info += 'SD: ' + \
        f"{', '.join(sd_unsaved)} <br> total " + \
        f'{len(sd_unsaved)} models not in modelzoo <br>'
    load_info += 'LoRA: ' + \
        f"{', '.join(lora_unsaved)} <br> total" + \
        f'{len(lora_unsaved)} lora not in modelzoo'
    load_info += '</div>'
    if unsaved_list != []:
        return load_info, gr.update(value='upload all models',
                                    variant='primary')
    else:
        return load_info, gr.update(value='detect all exists models',
                                    variant='secondary')


def download_by_link(model_link: str,
                     turn_page_switch: int,
                     model_type='checkpoints',
                     md5='',
                     filename=''):
    """
    This function downloads a model from the given link.

    Args:
        model_link: The link to the model.
        is_lora: A boolean value indicating whether the model is a LORa model.
        md5: The MD5 checksum of the model.

    Returns:
        A string indicating whether the download was successful.
    """
    global mz
    # Get the output directory.
    is_controlnet = model_type == 'ControlNet'
    is_lora = model_type == 'Lora'
    if is_controlnet:
        output_dir = controlnet_dir
    elif is_lora:
        output_dir = lora_dir
    else:
        output_dir = stable_diffusion_dir
    model_tags = ['safetensors'] if not is_lora else ['lora']
    # Create a message to display at the top of the page.
    load_info = "download by model link: "

    # Get the filename from the link.
    if filename == '':
        load_info += 'please input filename<br />'
        return load_info, turn_page_switch
    if model_link == '':
        load_info += 'please input model_link<br />'
        return load_info, turn_page_switch
    if is_controlnet:
        if not filename.endswith(('.pth', '.bin', '.ckpt', '.pt')):
            load_info += 'please input controlnet filename with suffix<br />'
            return load_info, turn_page_switch
    else:
        if not filename.endswith(('.safetensors', '.ckpt', '.bin')):
            load_info += 'please input filename with suffix<br />'
            return load_info, turn_page_switch
    try:
        res = requests.get(model_link, stream=True, timeout=5)
    except (requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            urllib3.exceptions.ReadTimeoutError, TimeoutError):
        load_info += 'timeout, suggest copy by oss to filebrowser<br />'
        return load_info, turn_page_switch
    except (requests.exceptions.MissingSchema):
        load_info += 'please input valid model link<br />'
        return load_info, turn_page_switch

    # Check the response status code.
    if res.status_code == 404 or res.status_code == 500:
        load_info += 'download failed<br />'
        return load_info, turn_page_switch
    target_model = os.path.join(output_dir, filename)
    f = open(target_model, 'wb')

    # If an MD5 checksum is provided,
    # check the checksum of the downloaded file.
    if md5 != '':
        m = hashlib.md5()
    for chunk in res.iter_content(chunk_size=512):
        if chunk:
            f.write(chunk)
        if md5 != '':
            m.update(chunk)
    f.close()

    # Check if the MD5 checksums match.
    result = True
    if md5 != '':
        result = m.hexdigest() == md5

    if result:
        if not is_controlnet:
            mz.create_model(target_model, filename, model_tags=model_tags)
        load_info += 'download {} successfully<br />'.format(filename)
    else:
        load_info += 'download {} failed<br />'.format(filename)

    return load_info, -turn_page_switch


def download_public_cache(models_selected, model_type, bool_download_public):
    global public_cache_dir, user_data_dir, public_data_dir, mz, download_for_public, show_public_models
    load_info = "download from public cache: "
    models_selected = set(json.loads(models_selected))
    existed_models = list()
    success_models = list()
    model_tags = list()

    if model_type == 'Stable-diffusion Checkpoints':
        model_type = 'Stable-diffusion'
    src_dir = os.path.join(public_cache_dir, model_type)
    tgt_dir = os.path.join(user_data_dir, model_type)
    if model_type in download_for_public or bool_download_public:
        tgt_dir = os.path.join(public_data_dir, model_type)
    os.makedirs(tgt_dir, exist_ok=True)

    if model_type is not None:
        model_tags.append(convert_model_type(model_type))

    for model in models_selected:
        if model not in show_public_models:
            continue
        source_model = os.path.join(src_dir, model)
        target_model = os.path.join(tgt_dir, model)

        if os.path.exists(target_model):
            existed_models.append(model)
            continue

        # if Stable-diffusion or Lora dir
        # copy preview image and civitai info.
        if model_type in ('Stable-diffusion', 'Lora'):
            img, info = correlated_info(model)
            try:
                shutil.copy(os.path.join(src_dir, img), os.path.join(tgt_dir, img))
                shutil.copy(os.path.join(src_dir, info), os.path.join(tgt_dir, info))
            except:
                print("copy error.")

        if model_type in ('annotator'):
            shutil.copytree(source_model, target_model)
        else:
            shutil.copy(source_model, target_model)
        mz.create_model(target_model, model, model_tags=model_tags)
        success_models.append(model)
        # print(f"copy from {source_model} to {target_model}")

    if success_models != []:
        load_info += f'download {", ".join(success_models)} success</br>'
    if existed_models != []:
        load_info += f'{", ".join(existed_models)} already exists in models</br>'

    return load_info


def download_api(models_selected, model_link: str,
    turn_page_switch: int,
    model_type='Stable-diffusion Checkpoints',
    md5='',
    filename='',
    download_public=False,
):
    load_info = "<div style='color:#111' align='center'>"

    load_info += download_public_cache(models_selected, model_type, download_public)
    info, turn_page_switch = download_by_link(model_link, turn_page_switch, model_type, md5, filename)
    load_info += info

    load_info += '</div>'

    return load_info, turn_page_switch


def public_cache(file_type: str):
    global public_cache_dir, show_public_models
    # check if --data-dir is delivered
    if file_type == "Stable-diffusion Checkpoints":
        file_type = "Stable-diffusion"
    # source_model_dir = os.path.join(data_dir, file_type)
    source_model_dir = os.path.join(public_cache_dir, file_type)
    show_public_models = list()

    code = f"""<!-- {time.time()} -->
    <div id="table_div">
    <table id="public_cache"">
        <thead>
            <tr>
                <th>
                    <input type="checkbox" class="gr-check-radio gr-checkbox all_extensions_toggle" onchange="toggle_all_extensions(event)" />
                    <abbr>model name</abbr>
                </th>
                <th>update time</th>
                <th>file size</th>
            </tr>
        </thead>
        <tbody>
    """

    for model in os.listdir(source_model_dir):
        if model.endswith(unmodel_list):
            continue
        
        show_public_models.append(model)
        current_model_path = os.path.join(source_model_dir, model)

        code += f"""
            <tr>
                <td><label><input class="gr-check-radio gr-checkbox extension_toggle" type="checkbox" onchange="toggle_extension(event)" name="{model}"/>&nbsp;{model}</label></td>
                <td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.stat(current_model_path).st_mtime))}</td>
                <td>{convert_size(os.path.getsize(current_model_path))}</td>
            </tr>
    """

    code += """
        </tbody>
    </table>
    </div>
    """

    return code


def create_tab(tab: ModelZooBrowserTab, current_gr_tab: gr.Tab):
    global public_cache_dir
    others_dir = False
    download_ui = False
    standard_ui = True

    if tab.name == 'Upload Model':
        others_dir = True
        standard_ui = False
    elif tab.name == 'Download Model':
        standard_ui = False
        download_ui = True

    with gr.Row():
        with gr.Column(visible=standard_ui, scale=10):
            warning_box = gr.HTML('<p>&nbsp')
    # download model UI
    with gr.Blocks() as download_model_ui:
        with gr.Row(visible=download_ui):
            with gr.Row(scale=1):
                download_warning = gr.HTML()
            with gr.Row(scale=1):
                create_warning = gr.HTML()
            with gr.Row(scale=2):
                with gr.Column(scale=1):
                    checkbox_download_public = gr.Checkbox(value=False, label='download to public dir')
                with gr.Column(scale=2):
                    download_model_type_select = gr.Dropdown(
                        value='Stable-diffusion Checkpoints',
                        choices=['Stable-diffusion Checkpoints', 'Lora', 'ControlNet', 'VAE',
                                 'clip', 'annotator', 'Codeformer', 'ESRGAN', 'GFPGAN', 'hypernetworks',
                                 'LDSR', 'SwinIR', 'VAE-approx'],
                        label='file type')
                with gr.Column(scale=1):
                    md5_sum = gr.Textbox(placeholder='(optional)', label='md5 sum')
                with gr.Column(scale=1):
                    model_name = gr.Textbox(label='model name')
                with gr.Column(scale=1):
                    stable_diffusion_file = gr.Textbox(label='model link')
                with gr.Column(scale=1):
                    download_button = gr.Button(value='Download',
                                                variant='primary')
        
        if cmd_opts.public_cache or os.path.exists(public_cache_dir):
            download_model_ui.load(fn=public_cache, inputs=[download_model_type_select], outputs=[create_warning])

    # create model UI
    with gr.Row(visible=others_dir):
        all_model_warning = gr.HTML()
        all_model_commit_button = gr.Button(value='detect all exists models')

    # ModelZoo UI
    with gr.Row(visible=standard_ui,
                elem_id=f'{tab.base_tag}_modelzoo_browser'):
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row(elem_id=f'{tab.base_tag}_modelzoo' +
                                '_browser_gallery_controls',
                                visible=False):
                        with gr.Column(scale=2, min_width=20):
                            page_index = gr.Number(value=1, label='Page Index')
                    with gr.Row():
                        image_gallery = gr.Gallery(
                            show_label=True,
                            elem_id=f'{tab.base_tag}_modelzoo_browser_gallery'
                        ).style(grid=opts.modelzoo_browser_page_columns)
                    with gr.Row(visible=False) as prompt_info_panel:
                        prompt_sampler = gr.Textbox(value='',
                                                    interactive=False,
                                                    label='sampler')
                        prompt_seed = gr.Textbox(value='',
                                                 interactive=False,
                                                 label='seed')
                        prompt_cfg_scale = gr.Textbox(value='',
                                                      interactive=False,
                                                      label='cfg scale')
                        prompt_steps = gr.Textbox(value='',
                                                  interactive=False,
                                                  label='steps')
                    with gr.Row():
                        with gr.Column():
                            delete_num = gr.Number(
                                value=1,
                                interactive=True,
                                visible=False,
                                label='delete next',
                                elem_id=f'{tab.base_tag}_modelzoo' +
                                '_browser_del_num')
                        with gr.Column(scale=2):
                            delete = gr.Button(
                                'Delete(in modelzoo)',
                                elem_id=f'{tab.base_tag}_modelzoo' +
                                '_browser_del_img_btn')
                        with gr.Column(scale=1):
                            modelzoo_refresh_button = gr.Button(
                                'Refresh',
                                variant='primary'
                            )

                with gr.Column(scale=1):
                    with gr.Row(scale=0.5) as sort_panel:
                        sort_by = gr.Dropdown(value='similarity',
                                              choices=[
                                                  'similarity', 'model name',
                                                  'created date',
                                                  'updated date', 'version',
                                                  'author', 'dataset',
                                                  'sampler', 'cfg scale'
                                              ],
                                              label='Sort by')
                        sort_order = ToolButton(value=down_symbol)
                    with gr.Row():
                        case_insensitive = gr.Checkbox(
                            value=True, label='case insensitive')
                        accurate_match = gr.Checkbox(value=False,
                                                     label='accurate match')
                        topk = gr.Textbox(value='-1',
                                          visible=False,
                                          label='maxium model count')
                    with gr.Row():
                        keyword_search = gr.Textbox(value='',
                                                    label='model search')
                        search_model_type_select = gr.Dropdown(
                            value='all',
                            choices=['all', 'checkpoints', 'lora'],
                            label='model type')
                    with gr.Row(visible=False) as generation_info_panel:
                        img_file_info = gr.Textbox(label='Model Description',
                                                   interactive=False,
                                                   visible=False,
                                                   lines=6,
                                                   max_lines=10)
                        prompt_info = gr.Textbox(label='Prompt',
                                                 interactive=False,
                                                 lines=3,
                                                 max_lines=6)
                        negative_prompt_info = gr.Textbox(
                            label='Negative prompt',
                            interactive=False,
                            lines=3,
                            max_lines=6)
                    with gr.Row():
                        img_file_name = gr.Textbox(value='',
                                                   label='Model Name',
                                                   interactive=False)
                    with gr.Row():
                        load_model_button = gr.Button(value='load model')
                    with gr.Row(visible=False) as prompt_panel:
                        show_prompt_zoo = gr.Button(value='show prompt zoo')
                        with gr.Row(visible=False) as send_buttons_panel:
                            send_buttons = \
                                    modules.generation_parameters_copypaste \
                                    .create_buttons(
                                        ['txt2img', 'img2img', 'inpaint'])
                    with gr.Row():
                        collected_warning = gr.HTML()
                    # hidden components.
                    with gr.Row(visible=False):
                        visible_img_num = gr.Number()
                        tab_base_tag_box = gr.Textbox(tab.base_tag)
                        # Index of the selected image/model in the entire list.
                        image_index = gr.Textbox(
                            value=-1,
                            elem_id=f'{tab.base_tag}_modelzoo' +
                            '_browser_image_index')
                        # A hidden button to change the image_index
                        #   when clicking on the image in gallery.
                        set_index = gr.Button(
                            'set_index',
                            elem_id=f'{tab.base_tag}_modelzoo' +
                            '_browser_set_index')
                        filenames = gr.State([])
                        # Save the ids in modelmeta of the corresponding
                        #   hisotry_records in all currently visible modelzoo.
                        prompt_ids = gr.State([])
                        hidden = gr.Textbox()
                        # A trigger button used to determine
                        #   whether to select modified sdwebui in checkpoints.
                        select_hidden = gr.Number(value=1)
                        # button needed to save PIL for the current image,
                        #   jump to txt2img or img2img etc.
                        current_image = gr.Image(type='pil')
                        # Save all visible images and their labels.
                        image_page_list = gr.Textbox(
                            elem_id=f'{tab.base_tag}_modelzoo' +
                            '_browser_image_page_list')
                        # refresh button, which refreshes the current page
                        #   once the change is triggered.
                        turn_page_switch = gr.Number(value=1,
                                                     label='turn_page_switch')
                        # After clicking the image in gallery, if delete image
                        #   the current selected image needs update.
                        select_image_switch = gr.Number(value=1)
                        # download model ids
                        # selected_models = gr.State(elem_id="selected_models", value=[])
                        selected_models = gr.Text(elem_id="selected_models", value="")

    # Model Event
    search_model_type_select.change(lambda s: (-s),
                                    inputs=[turn_page_switch],
                                    outputs=[turn_page_switch])
    sort_by.change(lambda s: (-s),
                   inputs=[turn_page_switch],
                   outputs=[turn_page_switch])
    sort_order.click(fn=sort_order_flip,
                     inputs=[turn_page_switch, sort_order],
                     outputs=[page_index, turn_page_switch, sort_order])
    # Prompt Event

    # Common Event
    keyword_search.submit(lambda s: (-s),
                          inputs=[turn_page_switch],
                          outputs=[turn_page_switch])
    delete.click(delete_modelzoo_item,
                 inputs=[
                     delete_num, img_file_name, filenames, image_index,
                     visible_img_num, turn_page_switch, select_image_switch,
                     image_page_list, prompt_ids
                 ],
                 outputs=[
                     filenames, delete_num, turn_page_switch, visible_img_num,
                     image_gallery, select_image_switch, image_page_list,
                     warning_box, prompt_ids
                 ])
    load_model_button.click(
        fn=load_model,
        inputs=[img_file_name, select_hidden],
        outputs=[warning_box, select_hidden, img_file_name])
    show_prompt_zoo.click(
        fn=control_promptzoo,
        inputs=[img_file_name, turn_page_switch, prompt_ids],
        outputs=[
            image_gallery, prompt_panel, show_prompt_zoo, turn_page_switch,
            prompt_ids, prompt_info, negative_prompt_info, prompt_sampler,
            prompt_seed, prompt_cfg_scale, prompt_steps, hidden, warning_box,
            image_page_list, keyword_search, generation_info_panel,
            prompt_info_panel, search_model_type_select, send_buttons_panel,
            sort_panel
        ])
    modelzoo_refresh_button.click(
        fn=lambda s: (-s),
        inputs=[turn_page_switch],
        outputs=[turn_page_switch]
    )
    # Hidden Event
    select_image_switch.change(fn=None,
                               inputs=[tab_base_tag_box, image_index],
                               outputs=None,
                               _js='modelzoo_browser_select_image')
    turn_page_switch.change(fn=get_page_by_keyword,
                            inputs=[
                                filenames, keyword_search, sort_by, sort_order,
                                search_model_type_select, case_insensitive,
                                topk, accurate_match, img_file_name
                            ],
                            outputs=[
                                filenames, image_gallery, img_file_name,
                                img_file_info, visible_img_num, warning_box,
                                image_page_list, show_prompt_zoo, prompt_ids
                            ])
    turn_page_switch.change(fn=None,
                            inputs=[tab_base_tag_box],
                            outputs=None,
                            _js='modelzoo_browser_turnpage')
    select_hidden.change(fn=None,
                         inputs=[img_file_name],
                         outputs=None,
                         _js='modelzoo_browser_selectCheckpoint')
    set_index.click(view_modelzoo_item_info,
                    _js='modelzoo_browser_get_current_img',
                    inputs=[
                        tab_base_tag_box, image_index, filenames,
                        turn_page_switch, prompt_ids, img_file_name
                    ],
                    outputs=[
                        img_file_name, image_index, hidden, turn_page_switch,
                        current_image, img_file_info, prompt_info,
                        negative_prompt_info, prompt_sampler, prompt_seed,
                        prompt_cfg_scale, prompt_steps, prompt_panel
                    ])
    img_file_name.change(fn=lambda: '',
                         inputs=None,
                         outputs=[collected_warning])

    # Download Model Tab
    # download_button.click(fn=download_by_link,
    #                       inputs=[
    #                           stable_diffusion_file, turn_page_switch,
    #                           download_model_type_select, md5_sum, model_name
    #                       ],
    #                       outputs=[create_warning, turn_page_switch])
    # download_button.click(_js="models_selected", fn=download_public_cache, inputs=[selected_models], outputs=[download_warning])
    download_button.click(_js="models_selected", fn=download_api, inputs=[selected_models, stable_diffusion_file, turn_page_switch, download_model_type_select, md5_sum, model_name, checkbox_download_public], outputs=[download_warning, turn_page_switch])
    if cmd_opts.public_cache or os.path.exists(public_cache_dir):
        download_model_type_select.change(fn=public_cache, _js="refresh_models", 
            inputs=[download_model_type_select],
            outputs=[create_warning]
        )
    # Upload Model Tab
    all_model_commit_button.click(
        fn=upload_all_models,
        inputs=None,
        outputs=[all_model_warning, all_model_commit_button])

    try:
        modules.generation_parameters_copypaste.bind_buttons(
            send_buttons, current_image, img_file_info)
    except Exception:
        pass


def run_pnginfo(image):
    if image is None:
        return '', '', '', '', ''
    try:
        geninfo, items = images.read_info_from_image(image)
        items = {**{'parameters': geninfo}, **items}

        info = ''
        for key, text in items.items():
            info += f"""
                <div>
                <p><b>{plaintext_to_html(str(key))}</b></p>
                <p>{plaintext_to_html(str(text))}</p>
                </div>
                """.strip() + '\n'
    except UnidentifiedImageError as e:
        print(e)
        geninfo = None
        info = ''

    return geninfo


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as modelzoo_browser:
        with gr.Tabs(elem_id='modelzoo_browser_tabs_container'):
            for i, tab in enumerate(tabs_list):
                with gr.Tab(
                        tab.name,
                        elem_id=f'{tab.base_tag}_modelzoo_browser_container'
                ) as current_gr_tab:
                    with gr.Blocks(analytics_enabled=False):
                        create_tab(tab, current_gr_tab)
        gr.Checkbox(opts.modelzoo_browser_preload,
                    elem_id='modelzoo_browser_preload',
                    visible=False)
        gr.Textbox(','.join([tab.base_tag for tab in tabs_list]),
                   elem_id='modelzoo_browser_tab_base_tags_list',
                   visible=False)
        gr.Textbox(value=gr.__version__,
                   elem_id='modelzoo_browser_gradio_version',
                   visible=False)

    return (modelzoo_browser, 'Model Zoo', 'modelzoo_browser'),


def move_setting(cur_setting_name, old_setting_name, option_info, section,
                 added):
    try:
        old_value = shared.opts.__getattr__(old_setting_name)
    except AttributeError:
        old_value = None
    try:
        new_value = shared.opts.__getattr__(cur_setting_name)
    except AttributeError:
        new_value = None
    if old_value is not None and new_value is None:
        # Add new option
        shared.opts.add_option(
            cur_setting_name, shared.OptionInfo(*option_info, section=section))
        shared.opts.__setattr__(cur_setting_name, old_value)
        added = added + 1
        # Remove old option
        shared.opts.data.pop(old_setting_name, None)

    return added


def modelmeta_intersection(meta_list1, meta_list2):
    res = list()
    list1 = [m.id for m in meta_list1]
    for m in meta_list2:
        if m.id in list1:
            res.append(m)

    return res


def on_ui_settings():
    modelzoo_browser_options = [
        ('modelzoo_browser_hidden_components', None, [],
         'Select components to hide', DropdownMulti, lambda: {
             'choices': components_list
         }),
        ('modelzoo_browser_with_subdirs', 'images_history_with_subdirs', True,
         'Include images in sub directories'),
        ('modelzoo_browser_preload', 'images_history_preload', False,
         'Preload images at startup'),
        ('modelzoo_browser_copy_image', 'images_copy_image', False,
         'Move buttons copy instead of move'),
        ('modelzoo_browser_delete_message', 'images_delete_message', True,
         'Print image deletion messages to the console'),
        ('modelzoo_browser_txt_files', 'images_txt_files', True,
         'Move/Copy/Delete matching .txt files'),
        ('modelzoo_browser_logger_warning', 'images_logger_warning', False,
         'Print warning logs to the console'),
        ('modelzoo_browser_logger_debug', 'images_logger_debug', False,
         'Print debug logs to the console'),
        ('modelzoo_browser_scan_exif', 'images_scan_exif', True,
         'Scan Exif-/.txt-data (initially slower, ' +
         'but required for many features to work)'),
        ('modelzoo_browser_mod_shift', None, False,
         'Change CTRL keybindings to SHIFT'),
        ('modelzoo_browser_mod_ctrl_shift', None, False, 'or to CTRL+SHIFT'),
        ('modelzoo_browser_enable_maint', None, True,
         'Enable Maintenance tab'),
        ('modelzoo_browser_ranking_pnginfo', None, False,
         "Save ranking in image's pnginfo"),
        ('modelzoo_browser_page_columns', 'images_history_page_columns', 6,
         'Number of columns on the page'),
        ('modelzoo_browser_page_rows', 'images_history_page_rows', 6,
         'Number of rows on the page'),
        ('modelzoo_browser_pages_perload', 'images_history_pages_perload', 20,
         'Minimum number of pages per load'),
    ]

    section = ('modelzoo-browser', 'Model Zoo')
    # Move historic setting names to current names
    added = 0
    for cur_setting_name, old_setting_name, *option_info \
            in modelzoo_browser_options:
        if old_setting_name is not None:
            added = move_setting(cur_setting_name, old_setting_name,
                                 option_info, section, added)
    if added > 0:
        shared.opts.save(shared.config_filename)

    for cur_setting_name, _, *option_info in modelzoo_browser_options:
        shared.opts.add_option(
            cur_setting_name, shared.OptionInfo(*option_info, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
