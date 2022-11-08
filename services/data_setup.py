
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


class SetupData():
    """Creates training and testing dataloaders and datasets.

    Keyword arguments:
    train_dir -- Path to train directory.
    test_dir -- Path to test directory
    data_transform -- torchvision transforms to perform on training and testing data.
    batch_size -- number of samples per batch in each dataloader.
    num_workers  -- an integer for number of worker per dataloader.

    Return:
    A tuple of train_dataloder, test_dataloader, class_names.
    """

    def __init__(self,
                 train_dir: str,
                 test_dir: str,
                 batch_size: int = 32,
                 num_workers: int = os.cpu_count()) -> None:
        self.train_dir = str(train_dir)
        self.test_dir = str(test_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_transforms(self, train=False, test=False):
        """create_transforms create transforms for given split

        Args:
            train (bool, optional): if transform is for training dataset mark as true. Defaults to False.
            test (bool, optional): if transform is for testing dataset mark as true. Defaults to False.

        Returns:
            data_transform: transform for given split
        """
        if train:
            data_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()])
        elif test:
            data_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()])

        return data_transform

    def create_datasets(self):
        """create_datasets creates datasets for train and test directories.

        Returns:
            train_data, test_data: tuple of datasets for splits
        """
        train_transform = self.create_transforms(train=True)
        test_transform = self.create_transforms(test=True)

        train_data = datasets.ImageFolder(root=self.train_dir,
                                          transform=train_transform)

        test_data = datasets.ImageFolder(root=self.test_dir,
                                         transform=test_transform)
        return train_data, test_data

    def create_dataloaders(self):
        """create_dataloaders create dataloaders for train and test datasets.

        Returns:
            train_dataloader, test_dataloader, class_names: tuple for given split as well as the class_names of the dataset.
        """
        train_data, test_data = self.create_datasets()

        class_names = train_data.classes

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=True,
                                      pin_memory=True)

        test_dataloader = DataLoader(dataset=test_data,
                                     shuffle=False,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True)

        return train_dataloader, test_dataloader, class_names
