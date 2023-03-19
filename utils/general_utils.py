import os
from collections import defaultdict
import torch
import time
from datetime import datetime


""" This file contains general functions and definitions used across all the project. """

''' The output can be either directed to the default output (write_to_log == False) or to a log file
(write to log == True). To start writing to a log call the set_write_to_log file and provide the directory of the log
file. '''
write_to_log = False
log_fp = None

# The root of the project: Since we expect the code to start executing in the 'entry_points' directory, the root is one
# directory above
project_root_dir = '.'
# The name of the directory in which we keep trained models
models_dir = os.path.join(project_root_dir, 'models')
# The default name of a model (assigned to new models)
default_model_name = 'model'
# Model and config files suffixes
model_file_suffix = '.mdl'
config_file_suffix = '.cfg'


def set_write_to_log(output_dir):
    """ After calling this function, and file named 'log.txt' will be created in the provided output_dir, and the output
        will be redirected to this log. """
    global write_to_log
    global log_fp
    write_to_log = True
    log_path = os.path.join(project_root_dir, output_dir, 'log.txt')
    log_fp = open(log_path, 'w')


def log_print(class_name, indent, my_str):
    """ Print the provided string 'my_str', with the provided number of indents, with a prefix stating the name of
        class/function from which this string was printed. """
    if class_name == '':
        prefix = ''
    else:
        prefix = '[' + class_name + '] '
    full_str = '\t' * indent + prefix + my_str
    if write_to_log:
        log_fp.write(full_str + '\n')
        log_fp.flush()
    else:
        print(full_str, flush=True)


def get_timestamp_str():
    """ Get str describing current timestamp. Will be used as a directory name (Windows doesn't allow colon in a
        directory name, so we need to replace all colons. """
    return str(datetime.now()).replace(' ', '_').replace(':', '-')


def init_entry_point(should_write_to_log):
    """ Initialization for code execution:
        - Create a directory for this specific execution (using timestamp)
        - Set the relevant value to the write_to_log flag
        - Set the relevant language
    """
    timestamp = get_timestamp_str()
    os.mkdir(os.path.join(project_root_dir, timestamp))
    if should_write_to_log:
        set_write_to_log(timestamp)

    return timestamp


def generate_dataset(filepath, generation_func, *args):
    """ Check if the file in the provided filepath exists. Is it does- return its content. Otherwise- use the provided
        generation function, save its result and return it. """
    if os.path.exists(filepath):
        dataset = torch.load(filepath)
    else:
        dataset = generation_func(*args)
        torch.save(dataset, filepath)
    return dataset


def for_loop_with_reports(iterable, iterable_size, checkpoint_len, inner_impl, progress_report_func):
    """ Run over the provided iterable, for each element run the provided inner_impl function, and every checkpoint_len
        elements report progress using the provided progress_report_func. """
    checkpoint_time = time.time()

    for index, item in enumerate(iterable):
        should_print = False

        if index % checkpoint_len == 0:
            time_from_prev_checkpoint = time.time() - checkpoint_time
            progress_report_func(index, iterable_size, time_from_prev_checkpoint)
            checkpoint_time = time.time()
            should_print = True

        inner_impl(index, item, should_print)


# Aggregation functions
# This section contains function that aggregate labels.
# This is needed because each image has multiple captions in most of the dataset.
# If we compute some structural property of each caption (e.g., whether the main verb is passive), each image would have
# multiple labels (e.g., 3 captions would be active and 2 would be passive). If we want to give each image a single
# label, we use an aggregation function, that given a list of labels, produces a single label (e.g., in the case of
# passive, we can decide that if at least on caption was passive, the image is labeled as passive).

def at_least_one_agg_func(input_list):
    """ Returns 1 if at least one label in the input list is non zero. """
    if len(input_list) == 0:
        assert False

    return int(len([x for x in input_list if x != 0]) > 0)


def likelihood_agg_func(input_list):
    """ Returns the percentage of 1's. """
    if len(input_list) == 0:
        assert False

    return len([x for x in input_list if x == 1])/len(input_list)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator/denominator


def get_image_id_to_count(struct_data):
    """ Given a list of (image_id, val)- the struct_data input- where image ids are not unique and val is a binary
        value, get 2 mappings: one from image id to the number of its instances in the list, and one from image id to
        the number of its instances in the list where the val was 1.
    """
    image_id_to_count = defaultdict(int)
    image_id_to_pos_count = defaultdict(int)
    for image_id, expressing_prop in struct_data:
        image_id_to_count[image_id] += 1
        image_id_to_pos_count[image_id] += expressing_prop
    return image_id_to_count, image_id_to_pos_count


def get_image_id_to_prob(struct_data):
    """ Given a list of (image_id, val)- the struct_data input- where image ids are not unique and val is a binary
        value, get a mapping from image id to the fraction of its instances in the list where the val was 1.
    """
    image_id_to_count, image_id_to_pos_count = get_image_id_to_count(struct_data)
    image_id_to_prob = {x: image_id_to_pos_count[x] / image_id_to_count[x] for x in image_id_to_count.keys()}
    return image_id_to_prob


def is_property_implemented(language, struct_property):
    if struct_property in ['negation', 'passive', 'root_pos', 'transitivity'] and language == 'Japanese':
        return False
    # elif struct_property == 'transitivity' and language == 'Chinese':
    #     return False
    else:
        return True


def filter_single_annotated_images(struct_data):
    annotation_num, _ = get_image_id_to_count(struct_data)
    return [x for x in struct_data if annotation_num[x[0]] > 1]


def fleiss_kappa(struct_data):
    """ Fleiss Kappa: see https://en.wikipedia.org/wiki/Fleiss%27_kappa.
    In our case subjects are images and categories are 0 (property not expressed) or 1 (property expressed). """

    class_num = 2
    struct_data = filter_single_annotated_images(struct_data)
    rating_num = len(struct_data)
    if rating_num == 0:
        return 2
    subjects = list(set([x[0] for x in struct_data]))
    N = len(subjects)

    # 1. First calculate n_ij: the number of raters who assigned the i-th subject to the j-th category
    n_ij_dict = {}
    for image_id, val in struct_data:
        if image_id not in n_ij_dict:
            n_ij_dict[image_id] = [0] * 2
        n_ij_dict[image_id][val] += 1

    # Second, calculate p_j, the proportion of all assignments which were to the j-th category
    p_j = [(1/rating_num) * sum([x[j] for x in n_ij_dict.values()]) for j in range(class_num)]

    # Next, calculate the number of ratings per subject
    n_i = {x[0]: x[1][0] + x[1][1] for x in n_ij_dict.items()}

    # Now calculate P_i, the extent to which raters agree for the i-th subject
    P_i = [(1/(n_i[i]*(n_i[i] - 1)))*sum([n_ij_dict[i][j]*(n_ij_dict[i][j] - 1) for j in range(class_num)])
                for i in subjects]

    # Now compute Pbar, the mean of the P_i's
    Pbar = (1/N) * sum(P_i)

    # and Pbar_e which go into the formula for kappa
    Pbar_e = sum([p_j[j]**2 for j in range(class_num)])

    # Now compute kappa
    kappa = (Pbar - Pbar_e)/(1 - Pbar_e)
    return kappa
