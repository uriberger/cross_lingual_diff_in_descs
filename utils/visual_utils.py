from PIL import Image, ImageDraw

class VisualUtils:
    def __init__(self, dataset_builder):
        self.dataset_builder = dataset_builder
        self.path_finder = dataset_builder.get_image_path_finder()
        self.gt_class_data, self.gt_bbox_data = dataset_builder.get_gt_classes_bboxes_data()
        self.class_mapping = dataset_builder.get_class_mapping()

    def draw_image(self, image_id):
        image_path = self.path_finder(image_id)
        image_obj = Image.open(image_path)
        if image_id in self.gt_bbox_data:
            bboxes = self.gt_bbox_data[image_id]
            class_names = [self.class_mapping[x] for x in self.gt_class_data[image_id]]
            draw_obj = ImageDraw.Draw(image_obj)
            for bbox_ind in range(len(bboxes)):
                bbox = bboxes[bbox_ind]
                class_name = class_names[bbox_ind]
                draw_obj.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline='red')
                draw_obj.text((bbox[0], bbox[1]), str(bbox_ind) + '. ' + class_name, fill='red')
        image_obj.save(str(image_id) + '.jpg')
