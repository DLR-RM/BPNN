"""Implementation of the Washington University's RGB-D Object Dataset."""
import os
import re
from typing import Optional, Callable, Tuple, Any, Dict, List

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.mnist import download_and_extract_archive
from torchvision.datasets.utils import verify_str_arg


class WRGBD(VisionDataset):
    """`Washington University's RGB-D Object Dataset
    <http://https://rgbd-dataset.cs.washington.edu>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``WRGBD/processed/training.pt``
            and  ``WRGBD/processed/test.pt`` exist.
        categories: List of included categories. Default: all categories
        excluded_categories: List of excluded categories. Default no category
        split (string): One of {'train', 'val', 'test'}.
            Accordingly dataset is selected.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    url = "https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/rgbd-dataset.tar"

    split_list = ('train', 'val', 'test')

    all_categories = ['apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap',
                      'cell_phone', 'cereal_box', 'coffee_mug', 'comb', 'dry_battery', 'flashlight', 'food_bag',
                      'food_box', 'food_can', 'food_cup', 'food_jar', 'garlic', 'glue_stick', 'greens', 'hand_towel',
                      'instant_noodles', 'keyboard', 'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom',
                      'notebook', 'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato',
                      'rubber_eraser', 'scissors', 'shampoo', 'soda_can', 'sponge', 'stapler', 'tomato', 'toothbrush',
                      'toothpaste', 'water_bottle']

    def __init__(
            self,
            root: str,
            categories: Optional[List[str]] = None,
            excluded_categories: Optional[List[str]] = None,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.split_list)

        self.categories = [category for category in self.all_categories
                           if (categories is None or category in categories) and
                           (excluded_categories is None or category not in excluded_categories)]

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        instances = [(category, instance)
                     for category in self.categories
                     for instance in sorted(os.listdir(os.path.join(self.processed_folder, category)))]

        angle = '2' if self.split == 'test' else '[1|4]'
        pattern = re.compile(f'[A-z]+_[0-9]+_{angle}_[0-9]+_crop.png')

        self.data = []
        for category, instance in instances:
            for i, img_name in enumerate(sorted(os.listdir(os.path.join(self.processed_folder, category, instance)))):
                if pattern.match(img_name):
                    if self.split == 'test' \
                            or (self.split == 'train' and i % 5 != 0) \
                            or (self.split == 'val' and i % 5 == 0):
                        self.data.append((category, instance, img_name))

        self.classes = [instance for category, instance in instances]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data = self.data[index]
        category_name, target_name, img_name = data
        target = self.classes.index(target_name)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.processed_folder, *data))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        """Returns the folder to the raw data."""
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def processed_folder(self) -> str:
        """Returns the folder to the processed data."""
        return os.path.join(self.root, self.__class__.__name__, 'rgbd-dataset')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Returns the dict that maps the class to its index."""
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(os.path.exists(os.path.join(self.processed_folder, category)) for category in self.categories)

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        filename = self.url.rpartition('/')[2]
        download_and_extract_archive(self.url, download_root=self.raw_folder,
                                     extract_root=self.raw_folder, filename=filename, md5=None)

        print('Done!')

    def extra_repr(self) -> str:
        """Adds split to its representation."""
        return "Split: {}\nCategories: {}".format(self.split, self.categories)
