from torch.utils.data import Dataset, DataLoader, Subset
from lightning import LightningDataModule
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from PIL import Image
import random
import os



class INaturalistDataset(Dataset):
    """Custom dataset class for the iNaturalist dataset."""
    def __init__(self,
                 kind: str = 'train',
                 transform = None,
                 target_transform = None):
        
        self.kind = kind
        self.transform = transform
        self.target_transform = target_transform

        self.base_path = f'../../data/{kind}'

        self.img_classes = os.listdir(self.base_path)

        # Map class names to integer labels
        self.classes_to_label = {}
        # List of indices for images per class
        self.indices_by_class = []
        # List of image paths for given dataset kind
        self.img_paths = []
        # Corresponding labels for the images
        self.img_labels = []

        for i, img_class in enumerate(self.img_classes):
            # set the class name to its integer label
            self.classes_to_label[img_class] = i

            class_path = os.path.join(self.base_path, img_class)

            # Store indices of the image as they appear in the dataset
            # This is used to create a subset of the dataset
            # for each class
            class_indices: list[int] = []

            for img_name in os.listdir(class_path):
                if not img_name.endswith('.jpg'):
                    continue

                img_path = os.path.join(class_path, img_name)

                self.img_labels.append(i)
                self.img_paths.append(img_path)

                class_indices.append(len(self.img_labels) - 1)
            
            # indices_by_class[i] holds the indices of images in class i
            self.indices_by_class.append(class_indices)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""

        return len(self.img_labels)
    
    def __getitem__(self, idx: int) -> tuple[any,any]:
        """Return a sample and its label at the given index."""
        
        image = Image.open(self.img_paths[idx]).convert('RGB')
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

def train_valid_split(dataset: Dataset, valid_ratio = 0.2, shuffle = True, shuffle_seed = 42) -> tuple[Subset, Subset]:
    """ Splits the dataset into training and validation sets based on the provided ratio."""

    train_indices = [] # indices for training set
    valid_indices = [] # indices for validation set

    # Check if the dataset has the 'indices_by_class' attribute
    if not hasattr(dataset, 'indices_by_class'):
        raise AttributeError('Provided dataset does not have \'indices_by_class\' attribute.\nCannot proceed further.')
    
    # Check if proviced valid_ratio is in the range [0,1]
    assert (valid_ratio > 0 and valid_ratio < 1), 'validation data proportion should be in range [0,1]'
    
    if shuffle:
        random.seed(shuffle_seed)
    
    # Iterate through each class and split the indices with that class into training and validation indices
    for class_label_int, img_idx_list_for_that_class in enumerate(dataset.indices_by_class):

        # Number of images in the validation set for that class
        valid_class_imgs_count = round(valid_ratio * len(img_idx_list_for_that_class))

        # Do shuffling INSIDE the class
        if shuffle:
            random.shuffle(img_idx_list_for_that_class)
        
        # Add the first valid_class_imgs_count indices to the validation set
        valid_indices.extend(img_idx_list_for_that_class[:valid_class_imgs_count])

        # Add the rest of the indices to the training set
        train_indices.extend(img_idx_list_for_that_class[valid_class_imgs_count:])

    # Shuffle the training and validation indices AFTER the split
    if shuffle:
        random.shuffle(train_indices)
        random.shuffle(valid_indices)
    
    return Subset(dataset, train_indices), Subset(dataset, valid_indices)


class OneHotEncoder:
    """One-hot encoder for integer labels."""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        """Convert the label to one-hot encoding."""
        
        return F.one_hot(label, num_classes=self.num_classes).float()

class INaturalistDataloader(LightningDataModule):
    """Lightning dataloader for the iNaturalist dataset."""

    def __init__(self,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 valid_ratio: float = 0.2,
                 shuffle: bool = True,
                 shuffle_seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

    def setup(self, stage: str = 'fit') -> None:
        """Setup the dataset for training and validation."""

        # Define the image transform so that the batch is consistent
        self.transform = transforms.Compose([
            transforms.Resize(size=(400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ])

        # Define the target transform for one-hot encoding of integer labels
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
            OneHotEncoder(num_classes=len(os.listdir('../../data/train')))
        ])

        if stage == 'fit':
            # Create the dataset instances for training and validation
            self.dataset_train_valid = INaturalistDataset(kind='train', transform=self.transform, target_transform=self.target_transform)

            # Split the training dataset into training and validation sets
            self.dataset_train, self.dataset_valid = train_valid_split(self.dataset_train_valid,
                                                                    valid_ratio=self.valid_ratio,
                                                                    shuffle=self.shuffle,
                                                                    shuffle_seed=self.shuffle_seed)
        
        if stage == 'test' or stage is None:
            # Create the test dataset instance
            self.dataset_test = INaturalistDataset(kind='test', transform=self.transform, target_transform=self.target_transform)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""

        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""

        return DataLoader(self.dataset_valid, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""

        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)