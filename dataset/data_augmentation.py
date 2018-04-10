from imgaug import augmenters as iaa
from scipy import misc
import os
import cv2
import pickle

from awe_dataset import prepareDataset


class DatasetAugmentation(object):

    def __init__(self, data_file, dataset_dir_out, data_type="awe1", n_augment=10, n_random=None,
                 skip_gray=True, DEBUG=False):
        self.data_file = data_file
        self.dataset_dir_out = dataset_dir_out
        self.data_type = data_type
        self.n_augment = n_augment
        self.n_random = n_random
        self.skip_gray = skip_gray
        self.DEBUG = DEBUG
        self.seq = self.set_augmentations()

    def set_augmentations(self):
        # Here you can change augmentations
        if self.data_type == "awe1":
            return iaa.Sequential([
                iaa.SomeOf((1, 6), [
                    iaa.GaussianBlur(sigma=(0, 1)),
                    iaa.Add((-45, 45)),
                    iaa.Multiply((0.25, 1.5)),
                    iaa.ContrastNormalization((0.8, 1.2)),
                    iaa.Affine(rotate=(-25, 25)),
                    iaa.Affine(scale=(0.5, 1.2))
                ])
            ])

    def augment_data(self):
        train_file = []

        img_idx = 1
        for file in self.data_file:
            print("Processing file: {} [{}/{}]".format(file, img_idx, len(self.data_file)))
            img_idx += 1
            path_img = file
            basename, ext = os.path.basename(file).split(".")
            img = misc.imread(path_img)
            label = os.path.basename(os.path.dirname(path_img))

            dir_to_save = os.path.join(self.dataset_dir_out, label)
            if not os.path.exists(dir_to_save):
                os.makedirs(dir_to_save)

            # skip grayscale images
            if len(img.shape) < 3 and self.skip_gray:
                print("skipping grayscale image")
                continue

            for aug_idx in range(self.n_augment + 1):
                new_name = "{}_{}_{}.{}".format(label, basename, aug_idx, ext)

                # augment images and rectangles
                img_aug = self.seq.augment_image(img)

                if self.DEBUG:
                    cv2.imshow("Augmented dataset", img_aug)
                    cv2.waitKey(0)

                img_to_save = None
                if aug_idx == 0:
                    # save original image
                    img_to_save = img
                else:
                    # save augmented image
                    img_to_save = img_aug

                if not self.DEBUG:
                    misc.imsave(os.path.join(dir_to_save, new_name), img_to_save)
                    train_file.append(os.path.join(dir_to_save, new_name))

        cv2.destroyAllWindows()

        return train_file


if __name__ == "__main__":
    print("Starting data augmentation...")
    train_data, test_data = prepareDataset("dataset/awe/")

    # car plates
    dataset = DatasetAugmentation(train_data,
                                  "dataset/awe_train_aug",
                                  data_type="awe1",
                                  n_augment=100,
                                  DEBUG=False)

    train_data_aug = dataset.augment_data()

    print("Writing data files using pickle...")
    with open("dataset/train_data_aug.pkl", "wb") as f:
        pickle.dump(train_data_aug, f)

    with open("dataset/test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
