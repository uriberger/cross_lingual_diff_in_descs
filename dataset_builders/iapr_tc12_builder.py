import os
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


class IAPRTC12ImagePathFinder(ImagePathFinder):

    def __init__(self, images_dir_path):
        super(IAPRTC12ImagePathFinder, self).__init__()

        self.images_dir_path = images_dir_path

    def get_image_path(self, image_id):
        image_file_name = f'{image_id}.jpg'
        image_dir_ind = image_id // 1000
        image_dir_name = '{0:02d}'.format(image_dir_ind)
        image_path = os.path.join(self.images_dir_path, image_dir_name, image_file_name)

        return image_path


class IAPRTC12DatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the builder for the IAPR TC-12 dataset, described in the paper "The IAPR TC-12 Benchmark: A New
        Evaluation Resource for Visual Information Systems" by Grubinger et al.
    """

    def __init__(self, root_dir_path, language, struct_property, indent):
        super(IAPRTC12DatasetBuilder, self).__init__(
            root_dir_path, 'iaprtc12', language, struct_property, indent
        )

        if language == 'English':
            language_str = 'eng'
            self.default_encoding = None
        elif language == 'German':
            language_str = 'ger'
            self.default_encoding = 'utf-8'
        else:
            self.log_print('Only English or German for the IAPR TC-12 dataset, stopping!')
            assert False

        annotations_dir_name = f'annotations_complete_{language_str}'
        self.annotations_dir_path = os.path.join(self.root_dir_path, annotations_dir_name)
        self.images_dir_path = os.path.join(self.root_dir_path, 'images')

    def get_sample_data(self, file_path, encoding=None):
        if encoding is None:
            if self.default_encoding is None:
                fp = open(file_path, 'r')
            else:
                fp = open(file_path, 'r', encoding=self.default_encoding)
        else:
            fp = open(file_path, 'r', encoding=encoding)

        for line in fp:
            if '<DESCRIPTION>' in line:
                caption = line.split('<DESCRIPTION>')[1].split('</DESCRIPTION>')[0]
            if '<IMAGE>' in line:
                image_path = line.split('<IMAGE>')[1].split('</IMAGE>')[0]
                image_id = int(image_path.split('/')[-1].split('.')[0])

        fp.close()
        return caption, image_id

    def get_caption_data(self):
        caption_data = []
        for _, dir_names, _ in os.walk(self.annotations_dir_path):
            for dir_name in dir_names:
                cur_dir_path = os.path.join(self.annotations_dir_path, dir_name)
                for _, _, file_names in os.walk(cur_dir_path):
                    for file_name in file_names:
                        file_path = os.path.join(cur_dir_path, file_name)
                        try:
                            caption, image_id = self.get_sample_data(file_path)
                        except UnicodeDecodeError:
                            caption, image_id = self.get_sample_data(file_path, 'latin-1')
                        if len(caption) > 0:
                            caption_data.append({'image_id': image_id, 'caption': caption})

        return caption_data

    def get_gt_classes_data_internal(self):
        return None

    def get_gt_bboxes_data_internal(self):
        return None

    def get_class_mapping(self):
        return None

    def create_image_path_finder(self):
        return IAPRTC12ImagePathFinder(self.images_dir_path)
