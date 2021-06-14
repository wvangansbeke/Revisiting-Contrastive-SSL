"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""

from torchvision.datasets.vision import VisionDataset
from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    """ Generates a list of samples of a form (path_to_sample).
    Args:
        directory (str): root dataset directory
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.
    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
    Returns:
        List[str]: samples of a form (path_to_sample)
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    
    subdirs = [os.path.join(directory, o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory,o))] 
    instances = []

    if len(subdirs) == 0:
        print('No subdirs found in root folder - Look for files directory in root folder')
        subdirs = [directory]
    
    for subdir in sorted(subdirs):
        for fname in sorted(os.listdir(subdir)):
            path = os.path.join(subdir, fname)
            if is_valid_file(path):
                instances.append(path)        
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/folder_x/xxx.ext
        root/folder_x/xxz.ext
        root/folder_y/xxy.ext

        or

        root/xxx.ext
        root/xxz.ext
        root/xxy.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        samples (list): List of (sample path) strings
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=None)
        
        samples = self.make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    @staticmethod
    def make_dataset(
        directory: str,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, extensions=extensions, is_valid_file=is_valid_file)

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            an image
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/folder_x/xxx.png
        root/folder_x/xxz.png
        root/folder_y/xxy.png

        or

        root/xxx.png
        root/xxz.png
        root/xxy.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        imgs (list): List of (image path) strings
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


class ImageFile(VisionDataset):
    """An image data loader where the samples are arranged in a file like:

        folder1/file1.ext
        folder1/file2.ext
        folder2/file1.ext
        folder2/file2.ext

        The paths are defined relative to the root directory path. 

    Args:
        root (string): Root directory path.
        file_path (string): Path to file list.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
         

    Attributes:
        samples (list): List of (sample path) strings
    """
    def __init__(
            self,
            root: str,
            file_path: str,
            transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super(ImageFile, self).__init__(root, transform=transform, target_transform=None)
        with open(file_path, 'r') as f:
            samples = f.readlines()
        self.samples = [sample.strip() for sample in samples]
        self.loader = loader

    def __getitem__(self, index: int) -> Any:
        path = os.path.join(self.root, self.samples[index])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)
