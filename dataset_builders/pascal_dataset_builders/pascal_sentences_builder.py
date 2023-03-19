import os
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


MULT_FACT = 1000000


class PascalSentencesImagePathFinder(ImagePathFinder):

    def __init__(self, images_dir_path, class_mapping):
        super(PascalSentencesImagePathFinder, self).__init__()

        self.images_dir_path = images_dir_path
        self.class_mapping = class_mapping
        ''' We want the image id to be composed of the class index and the image serial number.
        Each image serial number is 6 digits, so if we'll multiply the class index by 1000000, there will be 6 digits
        left for the serial number of the image. For example, if the class index is 8 and the serial number is 123456,
        the image id will be 8 * 1000000 + 123456 = 8,123,456. '''
        self.mult_fact = MULT_FACT

    def get_image_path(self, image_id):
        image_serial_num = image_id % self.mult_fact
        class_ind = image_id // self.mult_fact
        image_file_name = '2008_' + '{0:06d}'.format(image_serial_num) + '.jpg'
        image_file_path = os.path.join(self.images_dir_path, self.class_mapping[class_ind], image_file_name)
        return image_file_path


class PascalSentencesDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the builder for the Pascal sentences dataset, described in the paper "Collecting Image Annotations
        Using Amazon’s Mechanical Turk" by Rashtchian et al.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(PascalSentencesDatasetBuilder, self).__init__(
            root_dir_path, 'pascal_sentences', 'English', struct_property, indent)

        self.sentences_dir_path = os.path.join(self.root_dir_path, 'sentence')
        self.images_dir_path = os.path.join(self.root_dir_path, 'dataset')

    @staticmethod
    def caption_file_name_to_image_id(file_name, class_ind):
        return PascalSentencesDatasetBuilder.file_name_to_image_id(file_name, class_ind, '.txt')

    @staticmethod
    def image_file_name_to_image_id(file_name, class_ind):
        return PascalSentencesDatasetBuilder.file_name_to_image_id(file_name, class_ind, '.jpg')

    @staticmethod
    def file_name_to_image_id(file_name, class_ind, suffix):
        id_prefix = class_ind * MULT_FACT
        image_serial_num = int(file_name.split('2008_')[1].split(suffix)[0])
        image_id = id_prefix + image_serial_num
        return image_id

    def get_caption_data(self):
        caption_data = []
        class_mapping = self.get_class_mapping()
        class_to_ind = {class_mapping[i]: i for i in range(len(class_mapping))}
        subdir_names = sorted(os.listdir(self.sentences_dir_path))
        for subdir_name in subdir_names:
            if subdir_name in class_to_ind:
                subdir_path = os.path.join(self.sentences_dir_path, subdir_name)
                class_ind = class_to_ind[subdir_name]
                file_names = sorted(os.listdir(subdir_path))
                for file_name in file_names:
                    image_id = self.caption_file_name_to_image_id(file_name, class_ind)
                    file_path = os.path.join(subdir_path, file_name)
                    with open(file_path, 'r') as fp:
                        for line in fp:
                            caption = line.strip()
                            caption_data.append({'image_id': image_id, 'caption': caption})

        return caption_data

    def get_gt_classes_data_internal(self):
        caption_data = self.get_caption_data()
        all_image_ids = list(set([x['image_id'] for x in caption_data]))
        gt_class_data = {image_id: [image_id // self.mult_fact] for image_id in all_image_ids}
        return gt_class_data

    def get_gt_bboxes_data_internal(self):
        return None

    def get_class_mapping(self):
        class_names = os.listdir(self.sentences_dir_path)
        class_names = [x for x in class_names if x != '.keep']
        class_names.sort()
        return class_names

    def create_image_path_finder(self):
        return PascalSentencesImagePathFinder(self.images_dir_path, self.get_class_mapping())
