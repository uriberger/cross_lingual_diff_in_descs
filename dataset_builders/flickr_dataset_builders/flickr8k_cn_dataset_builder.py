import os
from Levenshtein import distance
from collections import defaultdict
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.flickr_dataset_builders.flickr30k_dataset_builder import \
    Flickr30kDatasetBuilder


class Flickr8kCNDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the flickr8k-cn dataset, described in the paper 'Adding Chinese Captions
        to Images' by Li et al.
        This dataset is based on the Flickr8k dataset.
    """

    def __init__(self, root_dir_path, struct_property, translated, indent):
        translated_str = ''
        if translated:
            translated_str = '_translated'
        super(Flickr8kCNDatasetBuilder, self).__init__(
            root_dir_path, 'flickr8kcn' + translated_str, 'Chinese',
            struct_property, Flickr30kDatasetBuilder, 'flickr30', indent
        )

        self.translated = translated

        data_dir_name = 'data'
        if translated:
            caption_file_name = 'flickr8kzhmtest.captions.txt'
        else:
            caption_file_name = 'flickr8kzhc.caption.txt'
        en_caption_file_name = 'flickr8kenc.caption.txt'

        self.caption_file_path = os.path.join(root_dir_path, data_dir_name, caption_file_name)
        self.en_caption_file_path = os.path.join(root_dir_path, data_dir_name, en_caption_file_name)
    
    def get_flickr30_caption_id_mapping(self):
        caption_file_path = self.en_caption_file_path

        english_flickr30_caption_data = self.base_dataset_builder.get_caption_data()
        im_to_caps = defaultdict(list)
        for sample in english_flickr30_caption_data:
            im_to_caps[sample['image_id']].append({'caption': sample['caption'], 'caption_id': sample['caption_id']})

        flickr30_caption_id_mapping = {}
        with open(caption_file_path, 'r') as fp:
            line_ind = 0
            for line in fp:
                striped_line = line.strip()
                if len(striped_line) == 0:
                    continue
                line_parts = striped_line.split()
                image_id = int(line_parts[0].split('_')[0])
                caption = ' '.join(line_parts[1:])
                caption_ind = int(line_parts[0][-1])
                caption_list = im_to_caps[image_id]
                caption_list = [{'caption': x['caption'], 'caption_id': x['caption_id']} for x in caption_list]
                no_space_caption = caption.replace(' ', '')
                no_space_caption_list = [{'caption': x['caption'].replace(' ', ''), 'caption_id': x['caption_id']} for x in caption_list]
                found_list = [x for x in no_space_caption_list if x['caption'] == no_space_caption]
                if len(found_list) == 0:
                    dist_list = [distance(x['caption'], caption) for x in caption_list]
                    min_dist = min(dist_list)
                    found_inds = [i for i in range(len(dist_list)) if dist_list[i] == min_dist]
                    found_ind = found_inds[0]
                    caption_id = caption_list[found_ind]['caption_id']
                else:
                    caption_id = found_list[0]['caption_id']
                flickr30_caption_id_mapping[(image_id, caption_ind)] = caption_id
                line_ind += 1

        return flickr30_caption_id_mapping
    
    def get_caption_data(self):
        image_id_captions_pairs = []

        if self.translated:
            flickr30_caption_id_mapping = self.get_flickr30_caption_id_mapping()

        with open(self.caption_file_path, 'r', encoding='utf8') as fp:
            for line in fp:
                striped_line = line.strip()
                if len(striped_line) == 0:
                    continue
                line_parts = striped_line.split()
                image_id = int(line_parts[0].split('_')[0])
                caption = ' '.join(line_parts[1:])
                caption_ind = int(line_parts[0][-1])
                if self.translated:
                    caption_id = flickr30_caption_id_mapping[(image_id, caption_ind)]
                else:
                    caption_id = Flickr30kDatasetBuilder.image_id_to_caption_id(image_id, caption_ind)
                image_id_captions_pairs.append({'image_id': image_id, 'caption': caption, 'caption_id': caption_id})

        return image_id_captions_pairs
