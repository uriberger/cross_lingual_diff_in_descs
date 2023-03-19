import os
from utils.general_utils import generate_dataset
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.coco_dataset_builders.coco_dataset_builder import \
    CocoDatasetBuilder


class DeCocoDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the DeCOCO dataset, described in the paper 'Multimodal Pivots for Image
        Caption Translation' by Hitschler et al.
        This dataset is based on the COCO dataset.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(DeCocoDatasetBuilder, self).__init__(
            root_dir_path, 'de_coco_translated', 'German', struct_property, CocoDatasetBuilder, 'COCO', indent
        )

        self.caption_file_name_prefixes = ['dev', 'devtest', 'test']

        self.line_ind_to_image_caption_id_file_path = os.path.join(
            self.cached_dataset_files_dir,
            f'{self.name}_line_ind_to_image_caption_id'
        )

    def get_line_ind_to_image_caption_id_mappings_internal(self):
        """ For some reason, even though the README file of this dataset states that there supposed to be files
        indicating to which image id each line refers, there is none. So I need to do it manually by searching for the
        English caption in the original COCO dataset and using the fact that the English and German files are aligned.
        """
        known_mappings = {'dev': {}, 'devtest': {130: (471312, 260559)}, 'test': {15: (453695, 589332), 386: (553184, 760473), 498: (488853, 353211)}}

        english_coco_caption_data = self.base_dataset_builder.get_caption_data()
        line_ind_to_image_caption_id_mappings = {}
        for caption_file_name_prefix in self.caption_file_name_prefixes:
            line_ind_to_image_caption_id_mappings[caption_file_name_prefix] = []
            caption_file_name = caption_file_name_prefix + '.en'
            caption_file_path = os.path.join(self.root_dir_path, caption_file_name)
            line_ind = 0
            with open(caption_file_path, 'r', encoding='utf8') as caption_fp:
                for line in caption_fp:
                    if line_ind in known_mappings[caption_file_name_prefix]:
                        orig_image_id, caption_id = known_mappings[caption_file_name_prefix][line_ind]
                        image_id = self.base_dataset_builder.orig_to_new_image_id(orig_image_id, 'train')
                    else:
                        caption = line.strip()
                        english_coco_caption_samples = [x for x in english_coco_caption_data
                                                        if x['caption'] == caption]
                        if len(english_coco_caption_samples) == 0:
                            english_coco_caption_samples = [x for x in english_coco_caption_data
                                                            if x['caption'].strip() == caption]
                        if len(english_coco_caption_samples) == 0:
                            english_coco_caption_samples = [x for x in english_coco_caption_data
                                                            if caption in x['caption']
                                                            and len(caption) >= len(x['caption']) - 1]
                        assert len(english_coco_caption_samples) > 0
                        image_id = english_coco_caption_samples[0]['image_id']
                        caption_id = english_coco_caption_samples[0]['caption_id']
                    line_ind_to_image_caption_id_mappings[caption_file_name_prefix].append((image_id, caption_id))
                    line_ind += 1

        return line_ind_to_image_caption_id_mappings

    def get_line_ind_to_image_caption_id_mappings(self):
        return generate_dataset(self.line_ind_to_image_caption_id_file_path, self.get_line_ind_to_image_caption_id_mappings_internal)

    def get_caption_data(self):
        line_ind_to_image_caption_id_mappings = self.get_line_ind_to_image_caption_id_mappings()
        caption_data = []
        for caption_file_name_prefix in self.caption_file_name_prefixes:
            caption_file_name = caption_file_name_prefix + '.de'
            caption_file_path = os.path.join(self.root_dir_path, caption_file_name)
            line_ind_to_image_caption_id_mapping = line_ind_to_image_caption_id_mappings[caption_file_name_prefix]
            line_ind = 0
            with open(caption_file_path, 'r', encoding='utf8') as caption_fp:
                for line in caption_fp:
                    caption = line.strip()
                    image_id, caption_id = line_ind_to_image_caption_id_mapping[line_ind]
                    caption_data.append({'image_id': image_id, 'caption_id': caption_id, 'caption': caption})
                    line_ind += 1

        return caption_data
