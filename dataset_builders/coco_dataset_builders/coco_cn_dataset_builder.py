import os
from collections import defaultdict
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import \
    CocoDatasetBuilder


class CocoCNDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the COCO-CN dataset, described in the paper 'COCO-CN for Cross-Lingual
        Image Tagging, Captioning, and Retrieval' by Li et al.
        This dataset is based on the COCO dataset.

        translated: A flag indicating whether we should use translated captions or original ones.
    """

    def __init__(self, root_dir_path, struct_property, translated, indent):
        translated_str = ''
        if translated:
            translated_str = '_translated'
        super(CocoCNDatasetBuilder, self).__init__(root_dir_path, 'coco_cn' + translated_str, 'Chinese',
                                                   struct_property, CocoDatasetBuilder, 'COCO', indent)

        caption_file_name_prefix = 'imageid.'
        caption_file_name_suffix = '-caption.txt'

        if translated:
            caption_file_name_content = 'manually-translated'
        else:
            caption_file_name_content = 'human-written'
        captions_file_name = caption_file_name_prefix + \
                             caption_file_name_content + \
                             caption_file_name_suffix

        self.captions_file_path = os.path.join(root_dir_path, captions_file_name)

    @staticmethod
    def image_id_to_caption_id(image_id, caption_ind):
        return 10000000*caption_ind + image_id
    
    @staticmethod
    def caption_id_to_image_id(caption_id):
        image_id = caption_id % 10000000
        caption_ind = caption_id // 10000000
        return image_id, caption_ind
    
    def get_caption_data(self):
        caption_data = []
        external_caption_file_path = self.captions_file_path
        with open(external_caption_file_path, 'r', encoding='utf8') as caption_fp:
            for line in caption_fp:
                line = line.strip()
                line_parts = line.split()
                image_file_name = line_parts[0]

                # Check if current image is from the relevant data split
                image_file_name_parts = image_file_name.split('_')
                data_split_str = image_file_name_parts[1].split('2014')[0]
                assert data_split_str in ['train', 'val']

                caption = line_parts[-1]
                orig_image_id = int(image_file_name_parts[-1].split('#')[0])
                if data_split_str == 'train':
                    image_id = 1000000 + orig_image_id
                else:
                    image_id = 2000000 + orig_image_id

                cur_caption_ind = int(line_parts[0][-1])
                caption_id = self.image_id_to_caption_id(image_id, cur_caption_ind)

                caption_data.append({'caption': caption, 'image_id': image_id, 'caption_id': caption_id})
        return caption_data
