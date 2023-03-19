import os
from collections import defaultdict
import gzip
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.flickr_dataset_builders.flickr30k_dataset_builder import \
    Flickr30kDatasetBuilder


class Multi30kDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the multi30k dataset, described in the paper 'Multi30K: Multilingual
        English-German Image Descriptions' by Elliott et al.
        This dataset is based on the Flickr30k dataset.

        translated: A flag indicating whether we should use translated captions or original ones.
    """

    def __init__(self, root_dir_path, language, struct_property, translated, indent):
        translated_str = ''
        if translated:
            translated_str = '_translated'
        super(Multi30kDatasetBuilder, self).__init__(
            root_dir_path, 'multi30k' + translated_str, language,
            struct_property, Flickr30kDatasetBuilder, 'flickr30', indent
        )

        self.translated = translated

        data_dir_name = 'data'
        if translated:
            task_dir_name = 'task1'
        else:
            task_dir_name = 'task2'
        raw_dir_name = 'raw'
        image_splits_dir_name = 'image_splits'

        train_file_name_prefix = 'train'
        val_file_name_prefix = 'val'
        test_file_name_prefix = 'test_2016'
        if translated:
            test_file_name_prefix += '_flickr'

        if language not in ['German', 'French']:
            assert False
        if language == 'French' and (not translated):
            self.log_print('Only translated captions for French. Stopping!')
            assert False

        if language == 'German':
            caption_file_name_suffix = 'de.gz'
        elif language == 'French':
            caption_file_name_suffix = 'fr.gz'
        en_caption_file_name_suffix = 'en.gz'
        if translated:
            image_file_name_suffix = '.txt'
        else:
            image_file_name_suffix = '_images.txt'

        self.caption_inds = range(1, 6)

        if translated:
            train_caption_file_name = f'{train_file_name_prefix}.{caption_file_name_suffix}'
            val_caption_file_name = f'{val_file_name_prefix}.{caption_file_name_suffix}'
            test_caption_file_name = f'{test_file_name_prefix}.{caption_file_name_suffix}'
            en_train_caption_file_name = f'{train_file_name_prefix}.{en_caption_file_name_suffix}'
            en_val_caption_file_name = f'{val_file_name_prefix}.{en_caption_file_name_suffix}'
            en_test_caption_file_name = f'{test_file_name_prefix}.{en_caption_file_name_suffix}'
        else:
            train_caption_file_names = [f'{train_file_name_prefix}.{i}.{caption_file_name_suffix}'
                                        for i in self.caption_inds]
            val_caption_file_names = [f'{val_file_name_prefix}.{i}.{caption_file_name_suffix}'
                                      for i in self.caption_inds]
            test_caption_file_names = [f'{test_file_name_prefix}.{i}.{caption_file_name_suffix}'
                                       for i in self.caption_inds]

        train_image_file_name = f'{train_file_name_prefix}{image_file_name_suffix}'
        val_image_file_name = f'{val_file_name_prefix}{image_file_name_suffix}'
        test_image_file_name = f'{test_file_name_prefix}{image_file_name_suffix}'

        task_dir_path = os.path.join(root_dir_path, data_dir_name, task_dir_name)

        caption_dir_path = os.path.join(task_dir_path, raw_dir_name)
        if translated:
            self.train_caption_file_paths = [os.path.join(caption_dir_path, train_caption_file_name)]
            self.val_caption_file_paths = [os.path.join(caption_dir_path, val_caption_file_name)]
            self.test_caption_file_paths = [os.path.join(caption_dir_path, test_caption_file_name)]
            self.en_train_caption_file_path = os.path.join(caption_dir_path, en_train_caption_file_name)
            self.en_val_caption_file_path = os.path.join(caption_dir_path, en_val_caption_file_name)
            self.en_test_caption_file_path = os.path.join(caption_dir_path, en_test_caption_file_name)
        else:
            self.train_caption_file_paths = [os.path.join(caption_dir_path, train_caption_file_names[i])
                                             for i in range(len(self.caption_inds))]
            self.val_caption_file_paths = [os.path.join(caption_dir_path, val_caption_file_names[i])
                                           for i in range(len(self.caption_inds))]
            self.test_caption_file_paths = [os.path.join(caption_dir_path, test_caption_file_names[i])
                                            for i in range(len(self.caption_inds))]

        image_dir_path = os.path.join(task_dir_path, image_splits_dir_name)
        self.train_image_file_path = os.path.join(image_dir_path, train_image_file_name)
        self.val_image_file_path = os.path.join(image_dir_path, val_image_file_name)
        self.test_image_file_path = os.path.join(image_dir_path, test_image_file_name)

    def get_line_to_image_id(self, data_split_str):
        line_to_image_id = []
        if data_split_str == 'train':
            image_file_path = self.train_image_file_path
        elif data_split_str == 'val':
            image_file_path = self.val_image_file_path
        elif data_split_str == 'test':
            image_file_path = self.test_image_file_path
        with open(image_file_path, 'r') as fp:
            for image_file_name in fp:
                image_id = int(image_file_name.split('.')[0])
                line_to_image_id.append(image_id)

        return line_to_image_id
    
    def get_line_to_caption_id(self, data_split_str, line_to_image_id):
        if data_split_str == 'train':
            caption_file_path = self.en_train_caption_file_path
        elif data_split_str == 'val':
            caption_file_path = self.en_val_caption_file_path
        elif data_split_str == 'test':
            caption_file_path = self.en_test_caption_file_path

        split_to_known_mappings = {'train': {23647: 25016890643}, 'val': {}, 'test': {}}
        known_mappings = split_to_known_mappings[data_split_str]

        english_coco_caption_data = self.base_dataset_builder.get_caption_data()
        im_to_caps = defaultdict(list)
        for sample in english_coco_caption_data:
            im_to_caps[sample['image_id']].append({'caption': sample['caption'], 'caption_id': sample['caption_id']})

        line_to_caption_id = []
        with gzip.open(caption_file_path, 'r') as fp:
            line_ind = 0
            for line in fp:
                if line_ind in known_mappings:
                    caption_id = known_mappings[line_ind]
                else:
                    caption = line.strip().decode('utf-8').replace('&amp;', '&').replace(' ', '').lower()
                    image_id = line_to_image_id[line_ind]
                    caption_list = im_to_caps[image_id]
                    caption_list = [{'caption': x['caption'].replace(' ', '').lower(), 'caption_id': x['caption_id']} for x in caption_list]
                    caption_id = [x for x in caption_list if x['caption'] == caption][0]['caption_id']
                line_to_caption_id.append(caption_id)
                line_ind += 1

        return line_to_caption_id
    
    def get_caption_data_for_split(self, data_split_str):
        line_to_image_id = self.get_line_to_image_id(data_split_str)
        if self.translated:
            line_to_caption_id = self.get_line_to_caption_id(data_split_str, line_to_image_id)
        else:
            image_id_to_caption_count = defaultdict(int)
        
        image_id_captions_pairs = []
        if data_split_str == 'train':
            caption_file_paths = self.train_caption_file_paths
        elif data_split_str == 'val':
            caption_file_paths = self.val_caption_file_paths
        elif data_split_str == 'test':
            caption_file_paths = self.test_caption_file_paths
        for caption_file_path in caption_file_paths:
            # with open(caption_file_path, 'r', encoding='utf-8') as fp:
            with gzip.open(caption_file_path, 'r') as fp:
                line_ind = 0
                for line in fp:
                    caption = line.strip().decode('utf-8')
                    image_id = line_to_image_id[line_ind]
                    if self.translated:
                        caption_id = line_to_caption_id[line_ind]
                    else:
                        caption_ind = image_id_to_caption_count[image_id]
                        image_id_to_caption_count[image_id] += 1
                        caption_id = Flickr30kDatasetBuilder.image_id_to_caption_id(image_id, caption_ind)
                    image_id_captions_pairs.append({
                        'image_id': image_id, 'caption': caption, 'caption_id': caption_id
                        })
                    line_ind += 1

        return image_id_captions_pairs

    def get_caption_data(self):
        return \
            self.get_caption_data_for_split('train') + \
            self.get_caption_data_for_split('val') + \
            self.get_caption_data_for_split('test')
