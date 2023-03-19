import os
import json
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import \
    CocoDatasetBuilder


class YJCaptionsDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the YJ-captions dataset, described in the paper 'Cross-Lingual Image
        Caption Generation' by Miyazaki and Shimizu.
        This dataset is based on the COCO dataset.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(YJCaptionsDatasetBuilder, self).__init__(
            root_dir_path, 'YJCaptions', 'Japanese', struct_property, CocoDatasetBuilder, 'COCO', indent
        )

        caption_file_name = 'yjcaptions26k_clean.json'
        self.caption_file_path = os.path.join(root_dir_path, caption_file_name)

    def get_caption_data(self):
        with open(self.caption_file_path, 'r', encoding='utf8') as caption_fp:
            caption_data = json.load(caption_fp)['annotations']
        # All images in de_coco are from the COCO train split, so we should add the train ids prefix
        return [{'image_id': 1000000 + x['image_id'], 'caption': x['caption']} for x in caption_data]
