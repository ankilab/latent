# Matrix operations
import numpy as np
# Abstract class for Data Generator
# from keras.utils import Sequence
from tensorflow.python.keras.utils.data_utils import Sequence
# Image normalization
from keras.applications.imagenet_utils import preprocess_input
# Image augmentations
import albumentations as A
# Reading images
import imageio as io
# Shuffling images
import random
# Convert RGB data to grayscale
from skimage.color import rgb2gray
# Convert image to uint8
from skimage import img_as_ubyte
# Resize images for batch_training
from skimage.transform import resize


class DataGenerator512x256(Sequence):
    def __init__(self, image_ids, label_ids, batch_size, augment=True, 
        shuffle=True, seed=42, scale_0_1=False
    ):
        """Provides images per batch with augmentation
        
        Parameters
        ----------
        image_ids : list
            A list of the image path names
        label_ids : list
            A list of the label path names
        batch_size : int
            batch size for model
        augment : bool, optional
            if augmentation should be applied to images, by default True
        shuffle : bool, optional
            if image/label pairs should be shuffled, by default True
        
        Returns
        -------
        tuple
            Contains two numpy arrays,
            each of shape (batch_size, height, width, 1). """
            
        self.batch_size = batch_size
        self.augment = augment
        self.image_ids = image_ids
        self.label_ids = label_ids
        self.shuffle = shuffle
        self.aug = self._get_augmenter()
        self.seed = seed
        self.scale_0_1 = scale_0_1

        random.seed(self.seed)
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data
        
        Parameters
        ----------
        index : int
            batch index in image/label id list
        
        Returns
        -------
        tuple
            Contains two numpy arrays,
            each of shape (batch_size, height, width, 1). 
        """
        X_ids = self.image_ids[(index*self.batch_size):(index+1)*self.batch_size]
        Y_ids = self.label_ids[(index*self.batch_size):(index+1)*self.batch_size]
        X = []
        Y = []

        for X_id, Y_id in zip(X_ids, Y_ids):
            image = io.imread(X_id)
            mask =  io.imread(Y_id) / 255

            if image.ndim == 3:
                image = (rgb2gray(image)*255).astype(np.uint8)

            if mask.ndim == 3:
                mask = mask[..., 0]

            if self.augment:
                augmented = self.aug(image=image, mask=mask) #augment image
                image = augmented['image']
                mask = augmented['mask']

            if self.scale_0_1:
                image = (image.astype(np.float32) - image.min()) / (image.max() - image.min())

            else:
                image = preprocess_input(image, mode='tf')

            X.append(image)
            Y.append(np.round(mask)) # to ensure binary targets

        return np.expand_dims(np.asarray(X), axis=3), \
                np.expand_dims(np.asarray(Y), axis=3)

    def on_epoch_end(self):
        """Prepare next epoch"""
        # Shuffle images
        if self.shuffle:
            shuffled = list(zip(self.image_ids,self.label_ids))
            random.shuffle(shuffled)
            self.image_ids, self.label_ids = zip(*shuffled)

    def _get_augmenter(self):
        """Defines used augmentations"""
        aug = A.Compose([
            A.RandomBrightnessContrast(p=0.75),    
            A.RandomGamma(p=0.75), 
            # border_mode = cv2.BORDER_CONSTANT = 0
            A.Rotate(limit=30, border_mode=0, p=0.75),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.5),
            A.GaussNoise(p=0.5)], p=1)
        return aug
