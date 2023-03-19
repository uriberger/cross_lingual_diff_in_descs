import os
import abc
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder


class EnglishBasedDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the base class for dataset based on English datasets, i.e., the same images with captions in a different
        language.
    """

    def __init__(self, root_dir_path, name, language, struct_property,
                 base_dataset_class, base_dataset_name, indent):
        super(EnglishBasedDatasetBuilder, self).__init__(
            root_dir_path, name, language, struct_property, indent
        )

        # This dataset doesn't contain the images themselves- the images are in the COCO dataset
        base_dataset_root_path = os.path.join(self.root_dir_path, '..', base_dataset_name)
        self.base_dataset_builder = base_dataset_class(base_dataset_root_path, self.struct_property,
                                                       self.indent + 1)

    @abc.abstractmethod
    def get_caption_data(self):
        return

    def get_gt_classes_data_internal(self):
        return self.base_dataset_builder.get_gt_classes_data()

    def get_gt_bboxes_data_internal(self):
        return self.base_dataset_builder.get_gt_bboxes_data()

    def get_class_mapping(self):
        return self.base_dataset_builder.get_class_mapping()

    def create_image_path_finder(self):
        return self.base_dataset_builder.create_image_path_finder()
