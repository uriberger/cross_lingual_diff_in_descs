import os
import json
import torch
from dataset_builders.single_dataset_builders.external_dataset_builders.image_caption_dataset_builders.image_caption_dataset_builder import ImageCaptionDatasetBuilder
from dataset_builders.image_path_finder import ImagePathFinder


MULT_FACT = 1000000


class CocoImagePathFinder(ImagePathFinder):

    def __init__(self, train_images_dir_path, val_images_dir_path):
        super(CocoImagePathFinder, self).__init__()

        self.train_images_dir_path = train_images_dir_path
        self.val_images_dir_path = val_images_dir_path
        self.mult_fact = MULT_FACT

    def get_image_path(self, image_id):
        image_split = image_id // self.mult_fact
        if image_split == 1:
            image_split_str = 'train'
            images_dir_path = self.train_images_dir_path
        elif image_split == 2:
            image_split_str = 'val'
            images_dir_path = self.val_images_dir_path
        else:
            assert False

        image_serial_num = image_id % self.mult_fact
        image_file_name = 'COCO_' + image_split_str + '2014_000000' + '{0:06d}'.format(image_serial_num) + '.jpg'
        image_path = os.path.join(images_dir_path, image_file_name)

        return image_path


class CocoDatasetBuilder(ImageCaptionDatasetBuilder):
    """ This is the dataset builder class for the MSCOCO dataset, described in the paper
        'Microsoft COCO: Common Objects in Context' by Lin et al.
        Something weird about COCO: They published 3 splits: train, val, test, but they didn't provide labels for the
        test split. So we're going to ignore the test set.
    """

    def __init__(self, root_dir_path, struct_property, indent):
        super(CocoDatasetBuilder, self).__init__(root_dir_path, 'COCO', 'English', struct_property, indent)

        train_val_annotations_dir = 'train_val_annotations2014'

        train_captions_file_path_suffix = os.path.join(train_val_annotations_dir, 'captions_train2014.json')
        self.train_captions_file_path = os.path.join(root_dir_path, train_captions_file_path_suffix)
        val_captions_file_path_suffix = os.path.join(train_val_annotations_dir, 'captions_val2014.json')
        self.val_captions_file_path = os.path.join(root_dir_path, val_captions_file_path_suffix)

        self.train_bboxes_file_name = 'instances_train2014.json'
        self.train_bboxes_file_path = os.path.join(root_dir_path, train_val_annotations_dir,
                                                   self.train_bboxes_file_name)
        self.val_bboxes_file_name = 'instances_val2014.json'
        self.val_bboxes_file_path = os.path.join(root_dir_path, train_val_annotations_dir,
                                                 self.val_bboxes_file_name)

        self.train_images_dir_path = os.path.join(root_dir_path, 'train2014')
        self.val_images_dir_path = os.path.join(root_dir_path, 'val2014')

    @staticmethod
    def orig_to_new_image_id(orig_image_id, data_split_str):
        if data_split_str == 'train':
            return 1*MULT_FACT + orig_image_id
        elif data_split_str == 'val':
            return 2*MULT_FACT + orig_image_id

    def get_caption_data(self):
        orig_train_caption_fp = open(self.train_captions_file_path, 'r')
        orig_val_caption_fp = open(self.val_captions_file_path, 'r')
        orig_train_caption_data = json.load(orig_train_caption_fp)['annotations']
        orig_val_caption_data = json.load(orig_val_caption_fp)['annotations']
        orig_train_caption_fp.close()
        orig_val_caption_fp.close()

        caption_data = [{'image_id': self.orig_to_new_image_id(x['image_id'], 'train'), 'caption': x['caption'], 'caption_id': x['id']}
                        for x in orig_train_caption_data] + \
                       [{'image_id': self.orig_to_new_image_id(x['image_id'], 'val'), 'caption': x['caption'], 'caption_id': x['id']}
                        for x in orig_val_caption_data]

        return caption_data

    def get_gt_classes_data_internal(self):
        gt_classes_data, _ = self.get_gt_classes_bboxes_data()
        return gt_classes_data

    def get_gt_bboxes_data_internal(self):
        _, gt_bboxes_data = self.get_gt_classes_bboxes_data()
        return gt_bboxes_data

    def get_gt_classes_bboxes_data(self):
        if os.path.exists(self.gt_classes_data_file_path):
            return torch.load(self.gt_classes_data_file_path), torch.load(self.gt_bboxes_data_file_path)
        else:
            img_classes_dataset = {}
            img_bboxes_dataset = {}
            for data_split_str in ['train', 'val']:
                if data_split_str == 'train':
                    external_bboxes_filepath = self.train_bboxes_file_path
                elif data_split_str == 'val':
                    external_bboxes_filepath = self.val_bboxes_file_path
                bboxes_fp = open(external_bboxes_filepath, 'r')
                bboxes_data = json.load(bboxes_fp)

                category_id_to_class_id = {bboxes_data[u'categories'][x][u'id']: x for x in
                                           range(len(bboxes_data[u'categories']))}

                # Go over all the object annotations
                for bbox_annotation in bboxes_data[u'annotations']:
                    image_id = bbox_annotation[u'image_id']
                    new_image_id = self.orig_to_new_image_id(image_id, data_split_str)
                    if new_image_id not in img_classes_dataset:
                        img_classes_dataset[new_image_id] = []
                        img_bboxes_dataset[new_image_id] = []

                    # First, extract the bounding box
                    bbox = bbox_annotation[u'bbox']
                    xmin = int(bbox[0])
                    xmax = int(bbox[0] + bbox[2])
                    ymin = int(bbox[1])
                    ymax = int(bbox[1] + bbox[3])
                    trnsltd_bbox = [xmin, ymin, xmax, ymax]

                    # Next, extract the ground-truth class of this object
                    category_id = bbox_annotation[u'category_id']
                    class_id = category_id_to_class_id[category_id]

                    img_classes_dataset[new_image_id].append(class_id)
                    img_bboxes_dataset[new_image_id].append(trnsltd_bbox)

            torch.save(img_classes_dataset, self.gt_classes_data_file_path)
            torch.save(img_bboxes_dataset, self.gt_bboxes_data_file_path)

            return img_classes_dataset, img_bboxes_dataset

    def get_class_mapping(self):
        bbox_fp = open(self.train_bboxes_file_path, 'r')
        bbox_data = json.load(bbox_fp)

        category_id_to_class_id = {bbox_data[u'categories'][x][u'id']: x for x in range(len(bbox_data[u'categories']))}
        category_id_to_name = {x[u'id']: x[u'name'] for x in bbox_data[u'categories']}
        class_mapping = {category_id_to_class_id[x]: category_id_to_name[x] for x in category_id_to_class_id.keys()}

        return class_mapping

    def create_image_path_finder(self):
        return CocoImagePathFinder(self.train_images_dir_path, self.val_images_dir_path)
