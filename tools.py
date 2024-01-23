import os
import matplotlib.pylab as plt
import seaborn as sns
import json
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold

class IOTools:
    def __init__(self) -> None:
        pass

    def read_dataframe(self, fname):
        df = pd.read_csv(fname)
        return df

    def load_model(self, fname):
        model = keras.models.load_model(fname)
        return model

    def save_json(self, obj, fname):
        with open(fname, "w") as f:
            json.dump(obj, f, indent=2)

    def save_dataframe(self, df, fname):
        df.to_csv(fname)

    def save_history(self, history, fname):
        df = pd.DataFrame(history.history)
        df.to_csv(fname)

    def save_model(self, model, fname):
        model.save(fname)

class DrawTools:
    def __init__(self) -> None:
        plt.rcParams["font.family"] = "DejaVu Serif"   # 使用するフォント
        plt.rcParams["font.size"] = 20                 # 文字の大きさ

    # ベクトルの表示
    def draw_vectors(self, vector_dict, fname):
        # print(vector_dict)
        plt.figure()
        for key, point in vector_dict.items():
            plt.plot([0, point[0]], [0, point[1]])
            plt.text(point[0], point[1], s=key)

        plt.title("AMINO VECTOR")
        self.save(fname)

    def draw_history(self, df, label, fname, xrange=(), yrange=()):
        acc = df[label]
        val_acc = df["val_" + label]
        epochs = [i+1 for i in range(len(acc))]
        plt.figure()
        plt.plot(epochs, acc, label=label)
        plt.plot(epochs, val_acc, label="val_"+label)

        if xrange: plt.xlim(xrange)
        if yrange: plt.ylim(yrange)

        # plt.title(label)
        plt.xlabel("epochs", fontsize=18)
        plt.ylabel(label, fontsize=18)
        plt.legend(fontsize=18)
        plt.tick_params(labelsize=18)

        plt.tight_layout()

        # plt.show()
        self.save(fname)

    def draw_confusion_matrix(self, cm, norm, fname):

        columns = ["A", "B", "C", "D", "E"]
        df = pd.DataFrame({
            columns[0]: cm[:, 0],
            columns[1]: cm[:, 1],
            columns[2]: cm[:, 2],
            columns[3]: cm[:, 3],
            columns[4]: cm[:, 4],
        }, index=columns)

        if norm:
            sns.heatmap(df, annot=True, square=True, cbar=True, cmap='Blues', annot_kws={"fontsize": 12}, fmt='.3f')
        else:
            sns.heatmap(df, annot=True, square=True, cbar=True, cmap='Blues', fmt='d', annot_kws={"fontsize": 12})
        plt.yticks(rotation=0)
        plt.xlabel("Pre", fontsize=24, rotation=0)
        plt.ylabel("GT", fontsize=24)
        self.save(fname)

    # ファイルを出力
    def save(self, fname):
        plt.savefig(fname, transparent=True)
        self.clear()

    # ファイルを閉じる
    def clear(self):
        plt.cla()
        plt.clf()
        plt.close()

class MLTools:
    def generate_cross_index(self, df, labels):
        index = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for train_idx, val_idx in kf.split(df, labels):
            index.append((train_idx, val_idx))
        return index

class Tools(IOTools, DrawTools, MLTools):
    def __init__(self) -> None:
        super().__init__()

if __name__ == "__main__":
    tools = Tools()
    # tools.draw_history("accuracy", "/home/mizuno/DeepImFam/result/mizuta/PUNT030102QIAN880126/256/history.csv")
    # tools.draw_history("loss", "/home/mizuno/DeepImFam/result/mizuta/PUNT030102QIAN880126/256/history.csv")
