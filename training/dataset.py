import random
import pathlib
from PIL import Image
from jittor.dataset import Dataset, RandomSampler, SequentialSampler
from jittor import transform as transforms
from jittor import dataset as data
import lmdb
import cv2
import os, sys
import pickle
import string
import io
import numpy as np

# from torchvision import transforms


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(Dataset):
    def __init__(self, path, image_mode="L", transform=None, max_images=None):
        super().__init__(batch_size=1)
        path = pathlib.Path(path)
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )
        if max_images is None:
            self.files = files
        elif max_images < len(files):
            self.files = random.sample(files, max_images)
        else:
            print(
                f"max_images larger or equal to total number of files, use {len(files)} images instead."
            )
            self.files = files
        self.transform = transform
        self.image_mode = image_mode
        self.mean, self.std = [0.5], [0.5]
        self.set_attrs(total_len=len(self.files))

    def __getitem__(self, index):
        image_path = self.files[index]
        image = Image.open(image_path).convert(self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.files)


class LmdbDataset(Dataset):
    def __init__(self, path, image_mode="L", transform=None, max_images=None):
        # path = pathlib.Path(path)
        super().__init__(batch_size=1)
        self.transform = transform
        self.image_mode = image_mode

        self.env = lmdb.open(
            path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

        cache_file = "_cache_" + "".join(c for c in path if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [
                    key for key in txn.cursor().iternext(keys=True, values=False)
                ]
            pickle.dump(self.keys, open(cache_file, "wb"))
        self.set_attrs(total_len=self.length)

    def __getitem__(self, index: int):
        img = None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        while img is None:
            try:
                try:
                    img = cv2.imdecode(np.fromstring(imgbuf, dtype=np.uint8), 1)
                    if img is None:
                        raise IOError("cv2.imdecode failed")
                    img = img[:, :, ::-1]  # BGR => RGB
                except IOError:
                    img = np.asarray(Image.open(io.BytesIO(imgbuf)))
                crop = np.min(img.shape[:2])
                img = img[
                    (img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2,
                    (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2,
                ]
                img = Image.fromarray(img, "RGB")
                img = img.resize((256, 256), Image.ANTIALIAS)
            except:
                print(sys.exc_info()[1])

        """
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert(self.image_mode)
        crop = np.min(img.shape[:2])
        img = Image.fromarray(img, "RGB")
        img = img[
            (img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2,
            (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2,
        ]
        img = img.resize((256, 256), Image.ANTIALIAS)
        """
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.length


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def create_dataloader(data_dir, size, batch, img_channel=3):
    mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.ImageNormalize(mean, std),
            # transforms.Normalize(mean, std, inplace=True),
        ]
    )

    if img_channel == 1:
        image_mode = "L"
    elif img_channel == 3:
        image_mode = "RGB"
    else:
        raise ValueError("image channel should be 1 or 3, but got ", img_channel)

    dataset = ImagePathDataset(data_dir, image_mode, transform).set_attrs(
        batch_size=batch, drop_last=True
    )
    sampler = data_sampler(dataset, shuffle=True)
    # print("sketch_batch: {}".format(batch))
    # loader = data.DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True)
    # return loader, sampler
    return sampler, sampler


def create_lmdb_dataloader(data_dir, size, batch, img_channel=3):
    mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.ImageNormalize(mean, std),
            # transforms.Normalize(mean, std, inplace=True),
        ]
    )

    if img_channel == 1:
        image_mode = "L"
    elif img_channel == 3:
        image_mode = "RGB"
    else:
        raise ValueError("image channel should be 1 or 3, but got ", img_channel)

    dataset = LmdbDataset(data_dir, image_mode, transform).set_attrs(
        batch_size=batch, drop_last=True
    )
    sampler = data_sampler(dataset, shuffle=True)
    # print("image_batch: {}".format(batch))
    # loader = data.DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True)
    # return loader, sampler
    return sampler, sampler


def yield_data(loader, sampler, distributed=False):
    epoch = 0
    while True:
        if distributed:
            # TODO: distributed
            assert False
            # sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
