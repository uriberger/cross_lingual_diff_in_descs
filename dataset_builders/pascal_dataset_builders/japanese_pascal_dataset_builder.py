import os
import csv
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.english_dataset_based_dataset_builder import \
    EnglishBasedDatasetBuilder
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.pascal_dataset_builders.pascal_sentences_builder import \
    PascalSentencesDatasetBuilder


class JapanesePascalDatasetBuilder(EnglishBasedDatasetBuilder):
    """ This is the dataset builder class for the Japanese Pascal dataset, described in the paper 'Image-Mediated
        Learning for Zero-Shot Cross-Lingual Document Retrieval' by Funaki and Nakayama.
        This dataset is based on the Pascal Sentences dataset.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(JapanesePascalDatasetBuilder, self).__init__(
            root_dir_path, 'pascal_jp', 'Japanese', struct_property,
            PascalSentencesDatasetBuilder, 'pascal_sentences', indent
        )

        self.correspondence_file_name = 'correspondence.csv'

    def get_ind_to_image_path_mapping(self):
        correspondence_file_path = os.path.join(self.root_dir_path, self.correspondence_file_name)
        reader = csv.reader(open(correspondence_file_path, 'r'))
        res = {}
        for line in reader:
            res[int(line[0])] = line[1]

        return res

    def get_caption_data(self):
        ind_to_image_path_mapping = self.get_ind_to_image_path_mapping()
        image_id_captions_pairs = []
        file_names = sorted(os.listdir(self.root_dir_path))
        file_names = [file_name for file_name in file_names
                      if file_name not in [self.correspondence_file_name, 'README.txt']]
        class_mapping = self.get_class_mapping()
        class_to_ind = {class_mapping[i]: i for i in range(len(class_mapping))}
        for file_name in file_names:
            file_path = os.path.join(self.root_dir_path, file_name)
            file_ind = int(file_name.split('.txt')[0])

            # Parse image path to get the image id
            image_path = ind_to_image_path_mapping[file_ind]
            image_path_parts = image_path.split('/')
            class_name = image_path_parts[0]
            image_file_name = image_path_parts[1]
            image_id = self.base_dataset_builder.image_file_name_to_image_id(image_file_name, class_to_ind[class_name])

            # Open file to get captions
            with open(file_path, 'r', encoding='utf8') as fp:
                for line in fp:
                    caption = line.strip()
                    image_id_captions_pairs.append({'image_id': image_id, 'caption': caption})

        return image_id_captions_pairs
