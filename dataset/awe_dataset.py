from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from random import shuffle
import os
from scipy.misc import imread, imsave, toimage, imresize
import numpy as np
import matplotlib.pyplot as plt


def prepareDataset(dataset_path, split_ratio=0.2, ext="png"):
    train_data = []
    test_data = []
    for subdir, dirs, files in os.walk(dataset_path):
        all_data = []
        for file in files:
            if file.endswith(ext):
                all_data.append(os.path.join(subdir, file))

        shuffle(all_data)
        test_idx = int(split_ratio * len(all_data))
        test_data += all_data[0:test_idx]
        train_data += all_data[test_idx:]

    return train_data, test_data


def compute_normalization(train_data, size=(128, 128)):
    images = []
    for img_path in train_data:
        I = imread(img_path, mode="RGB")
        I = imresize(I, size)
        images.append(np.expand_dims(I / 255, 0))
    images = np.concatenate(tuple(images), 0)

    mean_image = np.mean(images, axis=0)
    std_image = np.std(images, axis=0)

    meanR = np.mean(images[:, :, :, 0].flatten())
    meanG = np.mean(images[:, :, :, 1].flatten())
    meanB = np.mean(images[:, :, :, 2].flatten())

    stdR = np.std(images[:, :, :, 0].flatten())
    stdG = np.std(images[:, :, :, 1].flatten())
    stdB = np.std(images[:, :, :, 2].flatten())

    return mean_image, std_image, (meanR, meanG, meanB), (stdR, stdG, stdB)


class AWEDataset(Dataset):

    def __init__(self, data, transforms=None):
        self.transforms = transforms
        self.data = data

    def __getitem__(self, index):
        img_path = self.data[index]
        img = imread(img_path, mode="RGB")
        img = toimage(img)

        if self.transforms is not None:
            img = self.transforms(img)

        label = int(os.path.basename(os.path.dirname(img_path)))

        return img, label - 1

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_data, test_data = prepareDataset("/Users/dejanstepec/awe_capsules/CapsNet-pytorch/dataset/awe")

    mean_img, std_img, mean_c, std_c = compute_normalization(train_data)
    print(mean_img.shape)
    print(std_img.shape)
    print(mean_c)
    print(std_c)

    transformations = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                                          transforms.Normalize(mean_c, std_c)])
    awe_dataset = AWEDataset(train_data, transformations)

    awe_dataset_loader = DataLoader(dataset=awe_dataset,
                                    batch_size=10,
                                    shuffle=False)

    for images, labels in awe_dataset_loader:
        pass

