import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def _load_shapes3d(data_dir):
    transform = transforms.Compose([
        transforms.Normalize((0.5028, 0.5788, 0.6033),
                             (0.3492, 0.4011, 0.4213)),
    ])
    split = 'composition'
    trainset = Shapes3DDataset(f'{data_dir}/shapes3d_{split}_train_images.npz',
                               f'{data_dir}/shapes3d_{split}_train_labels.npz',
                                                       transform=transform)
    testset = Shapes3DDataset(f'{data_dir}/shapes3d_{split}_test_images.npz',
                              f'{data_dir}/shapes3d_{split}_test_labels.npz',
                                                      transform=transform)
    return trainset, testset

def _load_shapes3d_vae(data_dir):
    transform = transforms.Compose([
        transforms.Normalize((0.5028, 0.5788, 0.6033),
                             (0.3492, 0.4011, 0.4213)),
    ])
    split = 'composition'
    trainset = Shapes3DLatentDataset(f'{data_dir}/shapes3d_{split}_train_vectors_beta_1.npy',
                               f'{data_dir}/shapes3d_{split}_train_labels.npz',
                                                       transform=transform)
    testset = Shapes3DLatentDataset(f'{data_dir}/shapes3d_{split}_test_vectors_beta_1.npy',
                              f'{data_dir}/shapes3d_{split}_test_labels.npz',
                                                      transform=transform)
    return trainset, testset

def _load_mpi3d_vae(data_dir):
    transform = transforms.Compose([
        transforms.Normalize((0.5028, 0.5788, 0.6033),
                             (0.3492, 0.4011, 0.4213)),
    ])
    split = 'composition'
    trainset = Shapes3DLatentDataset(f'{data_dir}/mpi3d_{split}_train_vectors.npy',
                               f'{data_dir}/mpi3d_{split}_train_labels.npz',
                                                       transform=transform)
    testset = Shapes3DLatentDataset(f'{data_dir}/mpi3d_{split}_test_vectors.npy',
                              f'{data_dir}/mpi3d_{split}_test_labels.npz',
                                                      transform=transform)
    return trainset, testset

def _load_shapes3d_det(data_dir):
    transform = transforms.Compose([
        transforms.Normalize((0.5028, 0.5788, 0.6033),
                             (0.3492, 0.4011, 0.4213)),
    ])
    split = 'composition'
    trainset = Shapes3DLatentDataset(f'{data_dir}/shapes3d_{split}_train_images_beta_deterministic.npy',
                               f'{data_dir}/shapes3d_{split}_train_labels.npz',
                                                       transform=transform)
    testset = Shapes3DLatentDataset(f'{data_dir}/shapes3d_{split}_test_images_beta_deterministic.npy',
                              f'{data_dir}/shapes3d_{split}_test_labels.npz',
                                                      transform=transform)
    return trainset, testset

def _load_dsprites(data_dir):
    # transform = transforms.Compose([
    #     transforms.Normalize((0.5028),
    #                          (0.3492)),
    # ])
    transform = torchvision.transforms.Lambda(lambda x : x*2 - 1)
    split = 'composition'
    trainset = Shapes3DDataset(f'{data_dir}/dsprites_{split}_train_images.npz',
                               f'{data_dir}/dsprites_{split}_train_labels.npz',
                                                       transform=transform)
    testset = Shapes3DDataset(f'{data_dir}/dsprites_{split}_test_images.npz',
                              f'{data_dir}/dsprites_{split}_test_labels.npz',
                                                      transform=transform)
    return trainset, testset


def _load_mpi3d(data_dir):
    transform = transforms.Compose([
        transforms.Normalize((0.5028, 0.5788, 0.6033),
                             (0.3492, 0.4011, 0.4213)),
    ])
    split = 'composition'
    trainset = Shapes3DDataset(f'{data_dir}/mpi3d_{split}_train_images.npz',
                               f'{data_dir}/mpi3d_{split}_train_labels.npz',
                               transform=transform)
    testset = Shapes3DDataset(f'{data_dir}/mpi3d_{split}_test_images.npz',
                              f'{data_dir}/mpi3d_{split}_test_labels.npz',
                              transform=transform)
    return trainset, testset


class Shapes3DDataset(Dataset):
    """
    floor hue: 10 values linearly spaced in [0, 1]
    wall hue: 10 values linearly spaced in [0, 1]
    object hue: 10 values linearly spaced in [0, 1]
    scale: 8 values linearly spaced in [0, 1]
    shape: 4 values in [0, 1, 2, 3]
    orientation: 15 values linearly spaced in [-30, 30]
    """

    def __init__(self, images_path, latents_path, transform=None):
        super().__init__()

        images_files = np.load(images_path)
        self.images = torch.from_numpy(images_files['arr_0'])
        self.images = self.images / 255

        latents_files = np.load(latents_path)
        self.latents = torch.from_numpy(latents_files['arr_0'])

        self.transform = transform

    def __getitem__(self, idx):
        image, latents = self.images[idx], self.latents[idx]
        if self.transform:
            image = self.transform(image)
        return image, latents

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return (
            [f'floor_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'wall_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'object_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'scale={idx:.1f}' for idx in np.linspace(0,1,8)]+
            [f'shape={idx}' for idx in range(4)]+
            [f'orientation={int(idx):}' for idx in np.linspace(-30,30,15)]
        )

    @staticmethod
    def task_class_to_str(task_idx, class_idx):
        task_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        classes_per_task = {
            'floor_hue'  : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'wall_hue'   : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'object_hue' : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'scale'      : [f'{idx:.1f}' for idx in np.linspace(0,1,8)],
            'shape'      : list(range(4)),
            'orientation': [f'{idx:.1f}' for idx in np.linspace(-30,30,15)],
        }
        task_name = task_names[task_idx]
        class_value = classes_per_task[task_name][class_idx]
        return f'{task_name}={class_value}'

    @property
    def n_classes_by_latent(self):
        max_value_cls_per_task = self.latents.max(dim=0).values
        n_cls_per_task = max_value_cls_per_task + 1
        return tuple(n_cls_per_task.tolist())


class Shapes3DLatentDataset(Dataset):
    """
    floor hue: 10 values linearly spaced in [0, 1]
    wall hue: 10 values linearly spaced in [0, 1]
    object hue: 10 values linearly spaced in [0, 1]
    scale: 8 values linearly spaced in [0, 1]
    shape: 4 values in [0, 1, 2, 3]
    orientation: 15 values linearly spaced in [-30, 30]
    """

    def __init__(self, images_path, latents_path, transform=None):
        super().__init__()

        images_files = np.load(images_path)
        latents_files = np.load(latents_path)
        self.latents = torch.from_numpy(latents_files['arr_0'])
        self.images = torch.from_numpy(images_files)

    def __getitem__(self, idx):
        image, latents = self.images[idx], self.latents[idx]
        return image, latents

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return (
            [f'floor_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'wall_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'object_hue={idx:.1f}' for idx in np.linspace(0,1,10)]+
            [f'scale={idx:.1f}' for idx in np.linspace(0,1,8)]+
            [f'shape={idx}' for idx in range(4)]+
            [f'orientation={int(idx):}' for idx in np.linspace(-30,30,15)]
        )

    @staticmethod
    def task_class_to_str(task_idx, class_idx):
        task_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        classes_per_task = {
            'floor_hue'  : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'wall_hue'   : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'object_hue' : [f'{idx:.1f}' for idx in np.linspace(0,1,10)],
            'scale'      : [f'{idx:.1f}' for idx in np.linspace(0,1,8)],
            'shape'      : list(range(4)),
            'orientation': [f'{idx:.1f}' for idx in np.linspace(-30,30,15)],
        }
        task_name = task_names[task_idx]
        class_value = classes_per_task[task_name][class_idx]
        return f'{task_name}={class_value}'

    @property
    def n_classes_by_latent(self):
        max_value_cls_per_task = self.latents.max(dim=0).values
        n_cls_per_task = max_value_cls_per_task + 1
        return tuple(n_cls_per_task.tolist())

class LatentImageDataset(Dataset):
    def __init__(self, latents, images):
        """
        Custom Dataset for pairing latent vectors with images.
        Args:
            latents (numpy.ndarray or torch.Tensor): Latent vectors.
            images (numpy.ndarray or torch.Tensor): Corresponding images.
        """
        assert len(latents) == len(images), "Latents and images must have the same length."
        self.latents = torch.tensor(latents, dtype=torch.float32)
        self.images = torch.tensor(images, dtype=torch.float32)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        """
        Returns a single pair of latent vector and image.
        Args:
            idx (int): Index of the data point.
        Returns:
            tuple: (latent_vector, image)
        """
        latent_vector = self.latents[idx]
        image = self.images[idx]
        return latent_vector, image