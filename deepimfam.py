import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from aaindex import aaindex1
import statistics
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import tensorflow as tf
<<<<<<< HEAD
from tensorflow import keras
from keras import layers, optimizers, regularizers
=======
import keras
from keras import optimizers, regularizers
>>>>>>> 0b02021e00a1254091f4a1c7a17abdc9c77bf0c0
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score
# from sklearn.utils import compute_sample_weight
from imblearn.over_sampling import RandomOverSampler
from setconfig import SetConfig
from tools import Tools
from imageDataFrameGenrator import ImageDataFrameGenerator

class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

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
        aaindex3_dict = aaindex1[self.aaindex3].values
        aaindex1_values = aaindex1_dict.values()
        aaindex2_values = aaindex2_dict.values()
        aaindex3_values = aaindex3_dict.values()

        aaindex1_mean = statistics.mean(aaindex1_values)
        aaindex1_std = statistics.stdev(aaindex1_values)
        aaindex2_mean = statistics.mean(aaindex2_values)
        aaindex2_std = statistics.stdev(aaindex2_values)
        aaindex3_mean = statistics.mean(aaindex3_values)
        aaindex3_std = statistics.stdev(aaindex3_values)

        self.aavector = {}
        for key in aaindex1_dict:
            if key == "-": continue
            self.aavector[key] = np.array([
                (aaindex1_dict[key] - aaindex1_mean) / aaindex1_std,
                (aaindex2_dict[key] - aaindex2_mean) / aaindex2_std,
                (aaindex3_dict[key] - aaindex3_mean) / aaindex3_std,
            ])

        # ベクトルの可視化
        # fname = os.path.join(self.result_aaindex, "vectors.pdf")
        # self.draw_vectors(self.aavector, fname=fname)

    def calc_coordinate(self):
        self.load_GPCR()
        self.define_aavector()

        num = 0
        for (subsubfamily, seq) in zip(self.labels, self.sequences):
            points = [[0.0, 0.0, 0.0]]
            x, y, z = 0.0, 0.0, 0.0
            for aa in seq:
                if not aa in self.aavector: continue
                x += self.aavector[aa][0]
                y += self.aavector[aa][1]
                z += self.aavector[aa][2]
                points.append([x, y, z])

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
            fname = os.path.join(self.data_img, str(i) + ".png")
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
            color_mode="rgb",
            x_col="img_path",
            y_col="family",
            batch_size=256
        )

        return image_generator.load()

    def train(self):
        (train_gen, test_gen) = self.load_images()

        model = keras.models.Sequential([
                keras.layers.Conv2D(16, (2, 2), input_shape=(self.FIGSIZE, self.FIGSIZE, 3), padding="same"),
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
                keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                keras.layers.Dropout(self.dropout_ratio),
                keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                keras.layers.Dropout(self.dropout_ratio),
                keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                keras.layers.Dropout(self.dropout_ratio),
                keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
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
            factor=0.1,
            patience=20,
            min_lr=self.min_learning_rate
        )

        # モデル
        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=80,
        )

        model.fit(
            train_gen,
            validation_data=test_gen,  # Use validation_data instead of validation_batch_size
            epochs=self.epochs,
            batch_size=self.batch_size,  # Specify an appropriate batch size
            callbacks=[early_stopping, reduce_lr],
        )

        history = model.history

        self.save_model(model, self.model_fname)
        self.save_history(history, self.history_fname)

    def plot_result(self):
        fname = os.path.join(self.result_size, "history.csv")
        df = self.read_dataframe(fname)
        fname = os.path.join(self.result_size, "accuracy.pdf")
        self.draw_history(df, "accuracy", fname, yrange=(.8, 1.))
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

    def kfold_crossvalidate(self):
        tools = Tools()
        df = pd.read_csv(self.images_path)
        index = tools.generate_cross_index(df=df["img_path"], labels=df["family"])
        cnt = 0
        for train_idx, val_idx in index:
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

            sampler = RandomOverSampler(random_state=42)
            train_df, _ = sampler.fit_resample(train_df, train_df["family"])

            image_gen = ImageDataGenerator(
                rescale=1/255.,
            )

            train_gen = image_gen.flow_from_dataframe(
                dataframe=train_df,
                directory=self.data_img,
                shuffle=True,
                seed=0,
                x_col="img_path",
                y_col="family",
                target_size=(256, 256),
                batch_size=self.batch_size,
                color_mode="rgb",
                class_mode="categorical"
            )

            val_gen = image_gen.flow_from_dataframe(
                dataframe=val_df,
                directory=self.data_img,
                shuffle=False,
                # seed=0,
                x_col="img_path",
                y_col="family",
                target_size=(256, 256),
                batch_size=self.batch_size,
                color_mode="rgb",
                class_mode="categorical"
            )

            model = keras.models.Sequential([
                    keras.layers.Reshape((self.FIGSIZE, self.FIGSIZE * 3, 1), input_shape=(self.FIGSIZE, self.FIGSIZE, 3)),
                    keras.layers.Conv2D(16, (2, 6), strides=(1, 3), padding="same"),
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
                    # keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                    keras.layers.Dense(32, activation="relu"),
                    keras.layers.Dropout(self.dropout_ratio),
                    # keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                    keras.layers.Dense(512, activation="relu"),
                    keras.layers.Dropout(self.dropout_ratio),
                    # keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
                    keras.layers.Dense(512, activation="relu"),
                    keras.layers.Dropout(self.dropout_ratio),
                    keras.layers.Dense(64, activation="relu"),
                    # keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
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
                factor=0.1,
                patience=20,
                min_lr=self.min_learning_rate
            )

            # モデル
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=80,
            )

            model.fit(
                train_gen,
                validation_data=val_gen,  # Use validation_data instead of validation_batch_size
                epochs=self.epochs,
                batch_size=self.batch_size,  # Specify an appropriate batch size
                callbacks=[early_stopping, reduce_lr],
            )

            history = model.history

            fname = os.path.join(self.result_size, str(cnt) + "_crossval_model.h5")
            self.save_model(model, fname)
            fname = os.path.join(self.result_size, str(cnt) + "_crossval_history.csv")
            self.save_history(history, fname)

            cnt += 1

    def plot_cross_result(self):
        plt.figure()
        for cnt in range(5):
            fname = os.path.join(self.result_size, str(cnt) + "_crossval_history.csv")
            df = pd.read_csv(fname)
            acc = df["val_loss"]
            epochs = [i+1 for i in range(len(acc))]
            plt.plot(epochs, acc, label=str(cnt))

        # plt.ylim((.8, 1.))
        plt.xlabel("epochs", fontsize=18)
        plt.ylabel("val_loss", fontsize=18)
        plt.legend(fontsize=18)
        plt.tick_params(labelsize=18)
        plt.tight_layout()

        fname = os.path.join(self.result_size, "crossval_loss.pdf")
        plt.savefig(fname, transparent=True)

    def calc_macrof1(self):
        cnt = 0
        df = pd.read_csv(self.images_path)
        index = self.generate_cross_index(df["img_path"], df["family"])
        scores = []
        for _, val_idx in index:
            val_df = df.iloc[val_idx]

            image_gen = ImageDataGenerator(
                rescale=1/255.,
            )

            val_gen = image_gen.flow_from_dataframe(
                dataframe=val_df,
                directory=self.data_img,
                shuffle=False,
                # seed=0,
                x_col="img_path",
                y_col="family",
                target_size=(256, 256),
                batch_size=self.batch_size,
                color_mode="rgb",
                class_mode="categorical"
            )

            fname = os.path.join(self.result_size, str(cnt) + "_crossval_model.h5")
            model = self.load_model(fname)
            pred = model.predict(val_gen)
            pred = np.argmax(pred, axis=1)
            print(pred)

            score = f1_score(val_gen.classes, pred, average="macro")
            print(score)

            scores.append(score)

            cnt += 1

        print("macro-f1:", sum(scores) / len(scores))

if __name__ == '__main__':
    deepimfam = DeepImFam()

    # print(deepimfam.images_path)

    # ベクトルを割り当てる
    # deepimfam.define_aavector()

    # 座標を計算する
    # deepimfam.calc_coordinate()

    # 画像のパスデータを作成
    # print(deepimfam.images_path)
    # deepimfam.make_image_info()

    # deepimfam.load_images() # 画像の読み込み

    # 学習
    # deepimfam.train()

    # 結果のプロット
<<<<<<< HEAD
    # deepimfam.plot_result()
    # deepimfam.make_confusion_matrix()

    # 交差検証
    # deepimfam.kfold_crossvalidate()
    deepimfam.plot_cross_result()

    # deepimfam.calc_macrof1()
=======
    deepimfam.plot_result()
    deepimfam.make_confusion_matrix()
>>>>>>> 0b02021e00a1254091f4a1c7a17abdc9c77bf0c0
