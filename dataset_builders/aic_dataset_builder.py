import os
import json
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


class AICImagePathFinder(ImagePathFinder):

    def __init__(self, train_images_dir_path, val_images_dir_path, new_to_orig_image_id, train_sample_num):
        super(AICImagePathFinder, self).__init__()

        self.train_images_dir_path = train_images_dir_path
        self.val_images_dir_path = val_images_dir_path
        self.new_to_orig_image_id = new_to_orig_image_id
        self.train_sample_num = train_sample_num

    def get_image_path(self, image_id):
        orig_image_id = self.new_to_orig_image_id[image_id]

        image_file_name = f'{hex(orig_image_id)[2:]}.jpg'
        if image_id < self.train_sample_num:
            images_dir_path = self.train_images_dir_path
        else:
            images_dir_path = self.val_images_dir_path
        image_path = os.path.join(images_dir_path, image_file_name)

        return image_path


class AIChallengerDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the builder for the AI-Challenger dataset, described in the paper "AI Challenger : A Large-scale Dataset
        for Going Deeper in Image Understanding" by Wu et al.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(AIChallengerDatasetBuilder, self).__init__(
            root_dir_path, 'ai_challenger', 'Chinese', struct_property, indent
        )

        train_dir_name = 'ai_challenger_caption_train_20170902'
        val_dir_name = 'ai_challenger_caption_validation_20170910'
        train_captions_file_name = 'caption_train_annotations_20170902.json'
        val_captions_file_name = 'caption_validation_annotations_20170910.json'
        train_images_dir_name = 'caption_train_images_20170902'
        val_images_dir_name = 'caption_validation_images_20170910'

        self.train_captions_file_path = os.path.join(root_dir_path, train_dir_name, train_captions_file_name)
        self.val_captions_file_path = os.path.join(root_dir_path, val_dir_name, val_captions_file_name)
        self.train_images_dir_path = os.path.join(root_dir_path, train_dir_name, train_images_dir_name)
        self.val_images_dir_path = os.path.join(root_dir_path, val_dir_name, val_images_dir_name)

        self.new_to_orig_image_id = self.generate_new_to_orig_image_id_dict()
        self.orig_to_new_image_id = {self.new_to_orig_image_id[i]: i for i in range(len(self.new_to_orig_image_id))}

    @staticmethod
    def file_name_to_orig_image_id(file_name):
        return int(file_name.split('.jpg')[0], 16)

    def file_name_to_image_id(self, file_name):
        return self.orig_to_new_image_id[self.file_name_to_orig_image_id(file_name)]

    def load_from_json(self):
        loaded_data = []
        for data_split_str in ['train', 'val']:
            if data_split_str == 'train':
                external_caption_file_path = self.train_captions_file_path
            elif data_split_str == 'val':
                external_caption_file_path = self.val_captions_file_path
            with open(external_caption_file_path, 'r') as caption_fp:
                loaded_data.append(json.load(caption_fp))
        return loaded_data

    def generate_new_to_orig_image_id_dict(self):
        loaded_data = self.load_from_json()
        self.train_sample_num = len(loaded_data[0])
        all_data = loaded_data[0] + loaded_data[1]
        return [self.file_name_to_orig_image_id(x['image_id']) for x in all_data]

    def get_caption_data(self):
        loaded_data = self.load_from_json()
        all_data = loaded_data[0] + loaded_data[1]
        caption_data = [x for outer in
                        [[{'image_id': self.file_name_to_image_id(y['image_id']), 'caption': z} for z in y['caption']]
                         for y in all_data]
                        for x in outer]
        caption_data = [x for x in caption_data if len(x['caption'].strip()) > 0]
        new_caption_data = []
        counter = 0
        for x in caption_data:
            x['caption_id'] = counter
            new_caption_data.append(x)
            counter += 1
        return new_caption_data

    def get_gt_classes_data_internal(self):
        return None

    def get_gt_bboxes_data_internal(self):
        return None

    def get_class_mapping(self):
        return None

    def create_image_path_finder(self):
        return AICImagePathFinder(self.train_images_dir_path, self.val_images_dir_path, self.new_to_orig_image_id,
                                  self.train_sample_num)
