import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from aaindex import aaindex1
import statistics
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix
from setconfig import SetConfig
from tools import Tools
from imageDataFrameGenrator import ImageDataFrameGenerator

class DeepImFam(SetConfig, Tools):
    def __init__(self) -> None:
        super().__init__()
        tf.keras.backend.clear_session()

    # GPCRを読み込む
    def load_GPCR(self):
        self.labels = []
        self.sequences = []
        with open(self.gpcr_train_path) as f:
            for l in f.readlines():
                subsubfam, aaseq = l.split()
                self.labels.append(subsubfam)
                self.sequences.append(aaseq)

        with open(self.gpcr_test_path) as f:
            for l in f.readlines():
                subsubfam, aaseq = l.split()
                self.labels.append(subsubfam)
                self.sequences.append(aaseq)

    # アミノ酸にベクトルを割り当てる
    def define_aavector(self):
        # AAindexの指標を使用
        aaindex1_dict = aaindex1[self.aaindex1].values
        aaindex2_dict = aaindex1[self.aaindex2].values
        aaindex1_values = aaindex1_dict.values()
        aaindex2_values = aaindex2_dict.values()

        aaindex1_mean = statistics.mean(aaindex1_values)
        aaindex1_std = statistics.stdev(aaindex1_values)
        aaindex2_mean = statistics.mean(aaindex2_values)
        aaindex2_std = statistics.stdev(aaindex2_values)

        self.aavector = {}
        for key in aaindex1_dict:
            if key == "-": continue
            self.aavector[key] = np.array([
                (aaindex1_dict[key] - aaindex1_mean) / aaindex1_std,
                (aaindex2_dict[key] - aaindex2_mean) / aaindex2_std,
            ])

        # ベクトルの可視化
        fname = os.path.join(self.result_aaindex, "vectors.pdf")
        self.draw_vectors(self.aavector, fname=fname)

    def calc_coordinate(self):
        self.load_GPCR()
        self.define_aavector()

        num = 0
        for (subsubfamily, seq) in zip(self.labels, self.sequences):
            points = [[0.0, 0.0]]
            x, y = 0.0, 0.0
            for aa in seq:
                if not aa in self.aavector: continue
                x += self.aavector[aa][0]
                y += self.aavector[aa][1]
                points.append([x, y])

            fname = os.path.join(self.data_dat, str(num) + ".dat")
            with open(fname, "w") as f:
                for point in points:
                    print(", ".join(map(str, point)), sep="\n", file=f)
            num += 1

    def load_trans(self):
        self.dict_subsubfamily = {}
        self.dict_subfamily = {}
        self.dict_family = {}

        with open(self.trans_path) as f:
            for l in f.readlines():
                label, subsubfamily, family, subfamily = l.split()
                self.dict_subsubfamily[label] = subsubfamily
                self.dict_subfamily[label] = subfamily
                self.dict_family[label] = family

    def make_image_info(self):
        self.load_GPCR()
        self.load_trans()

        family = []
        subfamily = []
        subsubfamily = []
        img_path = []

        for i, label in enumerate(self.labels):
            subsubfamily.append(self.dict_subsubfamily[label])
            subfamily.append(self.dict_subfamily[label])
            family.append(self.dict_family[label])
            fname = os.path.join(self.data_img, str(i) + ".bmp")
            img_path.append(fname)

        df = pd.DataFrame.from_dict({
            "subsubfamily": subsubfamily,
            "subfamily": subfamily,
            "family": family,
            "img_path": img_path
        })

        self.save_dataframe(df, self.images_path)

    def load_images(self):
        df = pd.read_csv(self.images_path)
        image_generator = ImageDataFrameGenerator(
            images_dir=self.data_img,
            df=df,
            figsize=self.FIGSIZE,
            color_mode="grayscale",
            x_col="img_path",
            y_col="family",
            batch_size=256
        )

        return image_generator.load()

    def train(self):
        (train_gen, test_gen) = self.load_images()

        model = keras.models.Sequential([
                keras.layers.Conv2D(16, (2, 2), padding="same", input_shape=(self.FIGSIZE, self.FIGSIZE, 1)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(16, (2, 2), padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(32, (2, 2), padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(32, (2, 2), padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (2, 2), padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (2, 2), padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(self.dropout_ratio),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dropout(self.dropout_ratio),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dropout(self.dropout_ratio),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(5, activation="softmax"),
            ])

        model.summary()

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.max_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=20,
                        min_lr=self.min_learning_rate
        )

        # モデル
        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=150,
        )

        model.fit(
            train_gen,
            validation_data=test_gen,  # Use validation_data instead of validation_batch_size
            epochs=self.epochs,
            batch_size=self.batch_size,  # Specify an appropriate batch size
            callbacks=[reduce_lr, early_stopping],
        )

        history = model.history

        self.save_model(model, self.model_fname)
        self.save_history(history, self.history_fname)

    def plot_result(self):
        fname = os.path.join(self.result_size, "history.csv")
        df = self.read_dataframe(fname)
        fname = os.path.join(self.result_size, "accuracy.pdf")
        self.draw_history(df, "accuracy", fname)
        fname = os.path.join(self.result_size, "loss.pdf")
        self.draw_history(df, "loss", fname)

    def make_confusion_matrix(self):
        (train_gen, test_gen) = self.load_images()
        model = self.load_model(self.model_fname)

        labels = np.array(test_gen.classes)
        predict = np.argmax(model.predict(test_gen), axis=1)
        cm = confusion_matrix(labels, predict)
        fname = os.path.join(self.result_size, "cm.pdf")
        self.draw_confusion_matrix(cm, False, fname)

        norm_cm = confusion_matrix(labels, predict, normalize="true")
        fname = os.path.join(self.result_size, "norm_cm.pdf")
        self.draw_confusion_matrix(norm_cm, True, fname)

if __name__ == '__main__':
    deepimfam = DeepImFam()

    # ベクトルを割り当てる
    # deepimfam.define_aavector()

    # 座標を計算する
    # deepimfam.calc_coordinate()

    # 画像のパスデータを作成
    deepimfam.make_image_info()

    # deepimfam.load_images() # 画像の読み込み

    # 学習
    deepimfam.train()

    # 結果のプロット
    # deepimfam.plot_result()
    # deepimfam.make_confusion_matrix()
