"""Dataloader fewshot learning. Bridge for uncertainty_baselines"""
import jax
import numpy as np
import tensorflow_datasets as tfds
import torch
import torchvision
import re
import os
import os.path
import pathlib
import pandas as pd
import json

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from clu import preprocess_spec
from uncertainty_baselines.baselines import input_utils, preprocess_utils, ood_utils
from typing import Any, Callable, List, Optional, Tuple, Union, Sequence, Dict
from PIL import Image


class FewShotDataCreatorTF:
    """Class for few-shot data creator using uncertainty_baselines.
    Bridging between uncertainty_baselines and pytorch.
    """
    def __init__(self, fewshot_config, batch_size=None, save_dir=None):
        self.shots = fewshot_config["shots"]
        self.l2_regs = fewshot_config["l2_regs"]
        batch_size = batch_size or fewshot_config.get("batch_size")  # bwd compat.
        self.batch_size = batch_size
        self.pp_tr = fewshot_config["pp_train"]
        self.pp_te = fewshot_config["pp_eval"]
        self.walk_first = fewshot_config["walk_first"]
        self._datasets = {}  # This will be our cache for lazy loading.
        self.prefix_main = fewshot_config.get("prefix_main", "a/")
        self.prefix_lvl1 = fewshot_config.get("prefix_lvl1", "z/")
        self.prefix_lvl2 = fewshot_config.get("prefix_lvl2", "zz/")
        self.seed = fewshot_config.get("seed", 0)
        self.ood_datasets = fewshot_config.ood_datasets
        self.save_dir = save_dir

    def _get_dataset(self, dataset, train_split, test_split):
        """Lazy-loads given dataset."""
        key = (dataset, train_split, test_split)
        try:
            return self._datasets[key]
        except KeyError:
            # prepare train data
            train_ds = input_utils.get_data(
                dataset=dataset,
                split=train_split,
                rng=jax.random.PRNGKey(self.seed),
                process_batch_size=self.batch_size,
                preprocess_fn=preprocess_spec.parse(spec=self.pp_tr, available_ops=preprocess_utils.all_ops())
            )
            # preapre test data
            test_ds = input_utils.get_data(
                dataset=dataset,
                split=test_split,
                rng=jax.random.PRNGKey(self.seed),
                process_batch_size=2,
                num_epochs=1,
                preprocess_fn=preprocess_spec.parse(spec=self.pp_tr, available_ops=preprocess_utils.all_ops())
            )
            num_classes = tfds.builder(dataset).info.features["label"].num_classes

        return train_ds, test_ds, num_classes

    def _testset_collect(self, ds):
        # return as tensors/numpy arrays
        inputs_list = []
        labels_list = []

        count = 0
        for batch in tfds.as_numpy(ds):
            inputs_list.append(batch['image'].squeeze())
            labels_list.append(batch['label'].squeeze())
            print("counts", count)
            print("Accumulating inputs:", type(batch['image']), batch['image'].shape)
            print("Accumulating labels:", type(batch['label']), batch['label'].shape)
            count = count + 1
        inputs = np.concatenate(inputs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return inputs, labels

    def _fewshot_collect_from_whole_dataset(self, ds, shot, num_classes, steps=3):
        """Collection of few shot data in a tensor."""
        # return as tensors/numpy arrays
        inputs_list = []
        labels_list = []

        # collect whole data set
        for batch, _ in zip(tfds.as_numpy(ds), range(steps)):
            inputs_list.append(batch['image'].squeeze())
            labels_list.append(batch['label'].squeeze())
        inputs = np.concatenate(inputs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # Collect where we have samples of which classes.
        class_indices = [np.where(labels == cls_i)[0] for cls_i in range(num_classes)]
        all_idx = [indices[:shot] for indices in class_indices]
        all_idx = np.concatenate(all_idx, axis=0)
        
        # Extract few shot samples TODO: add the check for the dimensions and throw an error!
        numpy_x = inputs[all_idx]
        numpy_y = labels[all_idx]

        # Get the validation set
        val_idx = [indices[shot+1:shot+shot] for indices in class_indices]
        val_idx = np.concatenate(val_idx, axis=0)
        val_numpy_x = inputs[val_idx]
        val_numpy_y = labels[val_idx]

        return numpy_x, numpy_y, val_numpy_x, val_numpy_y

    def fewshot_dataset_preparations(self, dataset, train_split, test_split):
        """Return torch dataset for training."""
        # get the dataset in tf
        train_ds, test_ds, num_classes = self._get_dataset(dataset, train_split, test_split)

        # get the few shot train data
        for shot in self.shots:
            train_x, train_y, val_numpy_x, val_numpy_y = self._fewshot_collect_from_whole_dataset(train_ds, shot, num_classes)
            # save the data in the correct format
            filename_train_x = self.save_dir + './' + dataset + '_' + train_split + '_' + str(shot) + '_' + 'input.npy'
            filename_train_y = self.save_dir + './' + dataset + '_' + train_split + '_' + str(shot) + '_' + 'label.npy'
            filename_val_x = self.save_dir + './' + dataset + '_validation_' + str(shot) + '_' + 'input.npy'
            filename_val_y = self.save_dir + './' + dataset + '_validation_' + str(shot) + '_' + 'label.npy'
            np.save(filename_train_x, train_x)
            np.save(filename_train_y, train_y)
            np.save(filename_val_x, val_numpy_x)
            np.save(filename_val_y, val_numpy_y)

        # now getting the test and saving data
        test_x, text_y = self._testset_collect(test_ds)
        filename_test_x = self.save_dir + './' + dataset + '_' + test_split + '_' + 'input.npy'
        filename_test_y = self.save_dir + './' + dataset + '_' + test_split + '_' + 'label.npy'
        np.save(filename_test_x, test_x)
        np.save(filename_test_y, text_y)


class FewShotDataCreatorTorch:
    """Class for few-shot data creator using uncertainty_baselines.
    Bridging between uncertainty_baselines and pytorch.
    """
    def __init__(self, fewshot_config, batch_size=None, save_dir=None):
        self.shots = fewshot_config["shots"]
        self.l2_regs = fewshot_config["l2_regs"]
        batch_size = batch_size or fewshot_config.get("batch_size")  # bwd compat.
        self.batch_size = batch_size
        self.pp_tr = fewshot_config["pp_train"]
        self.pp_te = fewshot_config["pp_eval"]
        self.walk_first = fewshot_config["walk_first"]
        self._datasets = {}  # This will be our cache for lazy loading.
        self.prefix_main = fewshot_config.get("prefix_main", "a/")
        self.prefix_lvl1 = fewshot_config.get("prefix_lvl1", "z/")
        self.prefix_lvl2 = fewshot_config.get("prefix_lvl2", "zz/")
        self.seed = fewshot_config.get("seed", 0)
        self.ood_datasets = fewshot_config.ood_datasets
        self.save_dir = save_dir
        self.transforms = Compose([Resize([224, 224]),
                                  ToTensor(),
                                  Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                                  ])

    def _get_dataset(self, dataset, train_split, test_split, num_workers=8, pin_memory=False):
        """Lazy-loads given dataset."""
        key = (dataset, train_split, test_split)
        try:
            return self._datasets[key]
        except KeyError:
            if dataset == "caltech101":
                datasets = Caltech101(
                           root=self.save_dir + "../",
                           transform=self.transforms,
                           download=False
                           )
                num_train = len(datasets)
                indices = list(range(num_train))
                split = int(np.floor(0.1 * num_train))
                np.random.seed(0)
                np.random.shuffle(indices)
                train_idx, valid_idx = indices[split:], indices[:split]
                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)

                train_ds = torch.utils.data.DataLoader(
                    datasets, batch_size=self.batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                test_ds = torch.utils.data.DataLoader(
                    datasets, batch_size=self.batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
            elif dataset == "places365_small":
                train_ds = torchvision.datasets.Places365(
                    root=self.save_dir + "../" + dataset,
                    split='train-standard',
                    small=True,
                    download=True,
                    transform=self.transforms
                )
                test_ds = torchvision.datasets.Places365(
                    root=self.save_dir + "../" + dataset,
                    split='val',
                    small=True,
                    download=True,
                    transform=self.transforms
                )
                train_ds = torch.utils.data.DataLoader(
                    train_ds, batch_size=self.batch_size, 
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                test_ds = torch.utils.data.DataLoader(
                    test_ds, batch_size=self.batch_size, 
                    num_workers=num_workers, pin_memory=pin_memory,
                )
            elif dataset == "oxford_iiit_pet":
                train_ds = OxfordIIITPet(
                    root=self.save_dir + "../" + dataset,
                    split='trainval',
                    transform=self.transforms,
                    download=True,
                )
                test_ds = OxfordIIITPet(
                    root=self.save_dir + "../" + dataset,
                    split='test',
                    transform=self.transforms,
                    download=True,
                )
                train_ds = torch.utils.data.DataLoader(
                    train_ds, batch_size=self.batch_size, 
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                test_ds = torch.utils.data.DataLoader(
                    test_ds, batch_size=self.batch_size, 
                    num_workers=num_workers, pin_memory=pin_memory,
                )
            elif dataset == "imagenet2012_subset/10pct":
                imagenet_root=self.save_dir + "../" + dataset,
                imagenet_transform  = Compose([Resize(256),
                                               CenterCrop(224),
                                               ToTensor(),
                                               Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                               ])                            
                train_ds = ImageNetKaggle(root=imagenet_root,
                                          split='train',
                                          transform=imagenet_transform
                                         )
                test_ds = ImageNetKaggle(root=imagenet_root,
                                         split='val',
                                         transform=imagenet_transform
                                        )
                train_ds = torch.utils.data.DataLoader(
                    train_ds, batch_size=self.batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                test_ds = torch.utils.data.DataLoader(
                    test_ds, batch_size=self.batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
            elif dataset == "caltech_birds2011":
                train_ds = Cub2011(
                           root==self.save_dir + "../" + dataset,
                           transform=self.transforms,
                           train=True
                           )
                test_ds = Cub2011(
                           root==self.save_dir + "../" + dataset,
                           transform=self.transforms,
                           train=True
                           )
                train_ds = torch.utils.data.DataLoader(
                    train_ds, batch_size=self.batch_size
                )
                test_ds = torch.utils.data.DataLoader(
                    test_ds, batch_size=self.batch_size
                )
            else:
                raise AttributeError

            num_classes = tfds.builder(dataset).info.features["label"].num_classes
            
        return train_ds, test_ds, num_classes

    def _testset_collect(self, ds):
        # return as tensors/numpy arrays
        inputs_list = []
        labels_list = []

        count = 0
        for inputs, labels in ds:
            inputs_list.append(inputs.squeeze().permute(0, 2, 3, 1))
            labels_list.append(labels.squeeze())
            count = count + 1
        inputs = np.concatenate(inputs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return inputs, labels

    def _fewshot_collect_from_whole_dataset(self, ds, shot, num_classes, steps=3):
        """Collection of few shot data in a tensor."""
        # return as tensors/numpy arrays
        inputs_list = []
        labels_list = []

        # collect whole data set
        count = 1
        for inputs, labels in ds:
            inputs_list.append(inputs.squeeze().permute(0, 2, 3, 1))
            labels_list.append(labels.squeeze())
            if count == steps:
                print("breaking:", count)
                break
            count = count + 1
        inputs = np.concatenate(inputs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # Collect where we have samples of which classes.
        class_indices = [np.where(labels == cls_i)[0] for cls_i in range(num_classes)]
        all_idx = [indices[:shot] for indices in class_indices]
        all_idx = np.concatenate(all_idx, axis=0)
        
        # Extract few shot samples 
        numpy_x = inputs[all_idx]
        numpy_y = labels[all_idx]

        # Get the validation set
        val_idx = [indices[shot+1:shot+shot] for indices in class_indices]
        val_idx = np.concatenate(val_idx, axis=0)
        val_numpy_x = inputs[val_idx]
        val_numpy_y = labels[val_idx]

        return numpy_x, numpy_y, val_numpy_x, val_numpy_y

    def fewshot_dataset_preparations(self, dataset, train_split, test_split):
        """Return torch dataset for training."""
        # get the dataset in tf
        train_ds, test_ds, num_classes = self._get_dataset(dataset, train_split, test_split)

        # get the few shot train data
        for shot in self.shots:
            train_x, train_y, \
                val_numpy_x, val_numpy_y = self._fewshot_collect_from_whole_dataset(train_ds, shot, num_classes)
            # save the data in the correct format
            filename_train_x = self.save_dir + './' + dataset \
                + '_' + train_split + '_' + str(shot) + '_' + 'input.npy'
            filename_train_y = self.save_dir + './' + dataset \
                + '_' + train_split + '_' + str(shot) + '_' + 'label.npy'
            filename_val_x = self.save_dir + './' + dataset \
                + '_validation_' + str(shot) + '_' + 'input.npy'
            filename_val_y = self.save_dir + './' + dataset \
                + '_validation_' + str(shot) + '_' + 'label.npy'
            np.save(filename_train_x, train_x)
            np.save(filename_train_y, train_y)
            np.save(filename_val_x, val_numpy_x)
            np.save(filename_val_y, val_numpy_y)

        # now getting the test and saving data
        test_x, text_y = self._testset_collect(test_ds)
        filename_test_x = self.save_dir + './' + dataset \
            + '_' + test_split + '_' + 'input.npy'
        filename_test_y = self.save_dir + './' + dataset \
            + '_' + test_split + '_' + 'label.npy'
        np.save(filename_test_x, test_x)
        np.save(filename_test_y, text_y)


class Caltech101(VisionDataset):
    """`Caltech 101 <https://data.caltech.edu/records/20086>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        ).convert('RGB')

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            self.root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )

    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)


class OxfordIIITPet(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else: 
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), 
                             sep=' ',
                             names=['img_id', 'filepath']
                             )
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', 
                                         names=['img_id', 'target']
                                         )
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', 
                                       names=['img_id', 'is_training_img']
                                       )
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)    
    
    def __len__(self):
        return len(self.samples)    
    
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


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