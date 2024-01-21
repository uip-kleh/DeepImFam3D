import os, sys
sys.path.append(os.pardir)
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing.image import ImageDataGenerator


class ImageDataFrameGenerator:
    def __init__(
        self, images_dir, df, figsize, color_mode, x_col, y_col, class_mode="categorical", batch_size=256, seed=0
    ) -> None:
        self.images_dir = images_dir
        self.df = df
        self.figsize = figsize
        self.x_col = x_col
        self.y_col = y_col
        self.class_mode = class_mode
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.seed = seed

    def load(self):
        train_df, test_df = train_test_split(
            self.df, test_size=0.2, shuffle=True, random_state=1
        )

        sampler = RandomOverSampler(random_state=42)
        train_df, _ = sampler.fit_resample(train_df, train_df["family"])
        print(len(train_df))

        imageDataGenerator = ImageDataGenerator(
            # preprocessing_function=lambda img: img / 255.0,
            rescale=1/255.
        )

        trainDataFrameGenerator = imageDataGenerator.flow_from_dataframe(
            dataframe=train_df,
            directory=self.images_dir,
            shuffle=True,
            seed=self.seed,
            x_col=self.x_col,
            y_col=self.y_col,
            target_size=(self.figsize, self.figsize),
            batch_size=self.batch_size,
            color_mode=self.color_mode,
            class_mode=self.class_mode,
            subset="training",
        )

        testDataFrameGenerator = imageDataGenerator.flow_from_dataframe(
            dataframe=test_df,
            directory=self.images_dir,
            shuffle=False,
            seed=self.seed,
            x_col=self.x_col,
            y_col=self.y_col,
            target_size=(self.figsize, self.figsize),
            batch_size=self.batch_size,
            color_mode=self.color_mode,
            class_mode=self.class_mode,
        )

        return trainDataFrameGenerator, testDataFrameGenerator



if __name__ == "__main__":
    df = pd.read_csv("/home/mizuno/data/GPCR/cv_0/mizuta/images_path.csv")
