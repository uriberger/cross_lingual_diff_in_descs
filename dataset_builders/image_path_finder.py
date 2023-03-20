import abc


class ImagePathFinder:
    """ We have two dataset related objects: dataset builders and ImLingStructInfoDataset objects.
        The dataset builders save the data from external datasets in an organized manner to an external file.
        Each dataset builder has a 'build_dataset' function that creates a ImLingStructInfoDataset object. The constructors
        of these objects expect 2 things: the path to the file were we store the data, and an ImagePathFinder object that,
        given an image id, returns the path to the relevant image (since the stored data only holds image id and not the
        images themselves).
    """

    @abc.abstractmethod
    def get_image_path(self, image_id):
        return
