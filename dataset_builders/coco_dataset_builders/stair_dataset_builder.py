import os
import json
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import \
    CocoDatasetBuilder


class StairDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the STAIR-caption dataset, described in the paper 'STAIR Captions:
        Constructing a Large-Scale Japanese Image Caption Dataset' by Yoshikawa et al.
        This dataset is based on the COCO dataset.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(StairDatasetBuilder, self).__init__(
            root_dir_path, 'stair', 'Japanese', struct_property, CocoDatasetBuilder, 'COCO', indent
        )

        captions_file_prefix = 'stair_captions_v1.2'
        train_captions_file_name = f'{captions_file_prefix}_train.json'
        val_captions_file_name = f'{captions_file_prefix}_val.json'

        self.train_captions_file_path = os.path.join(root_dir_path, train_captions_file_name)
        self.val_captions_file_path = os.path.join(root_dir_path, val_captions_file_name)

    def get_caption_data(self):
        orig_train_caption_fp = open(self.train_captions_file_path, 'r', encoding='utf-8')
        orig_val_caption_fp = open(self.val_captions_file_path, 'r', encoding='utf-8')
        orig_train_caption_data = json.load(orig_train_caption_fp)['annotations']
        orig_val_caption_data = json.load(orig_val_caption_fp)['annotations']
        orig_train_caption_fp.close()
        orig_val_caption_fp.close()

        caption_data = [{'image_id': CocoDatasetBuilder.orig_to_new_image_id(x['image_id'], 'train'),
                         'caption': x['caption']} for x in orig_train_caption_data] + \
                       [{'image_id': CocoDatasetBuilder.orig_to_new_image_id(x['image_id'], 'val'),
                         'caption': x['caption']} for x in orig_val_caption_data]

        return caption_data
