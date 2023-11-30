import os
import matplotlib.pylab as plt
import seaborn as sns
import json
import pandas as pd
from tensorflow import keras

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
        pass

    # ベクトルの表示
    def draw_vectors(self, vector_dict, fname):
        # print(vector_dict)
        plt.figure()
        for key, point in vector_dict.items():
            plt.plot([0, point[0]], [0, point[1]])
            plt.text(point[0], point[1], s=key)

        plt.title("AMINO VECTOR")
        self.save(fname)

    def draw_history(self, df, label, fname):
        acc = df[label]
        val_acc = df["val_" + label]
        epochs = [i+1 for i in range(len(acc))]
        plt.figure()
        plt.plot(epochs, acc, label=label)
        plt.plot(epochs, val_acc, label="val_"+label)

        # plt.title(label)
        plt.xlabel("epochs")
        plt.ylabel(label)
        plt.legend()
        # plt.show()
        self.save(fname)

    def draw_confusion_matrix(self, cm, norm, fname):
        if norm:
            sns.heatmap(cm, annot=True, square=True, cbar=True, cmap='Blues')
        else:
            sns.heatmap(cm, annot=True, square=True, cbar=True, cmap='Blues', fmt='d')
        plt.yticks(rotation=0)
        plt.xlabel("Pre", fontsize=13, rotation=0)
        plt.ylabel("GT", fontsize=13)
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


class Tools(IOTools, DrawTools):
    def __init__(self) -> None:
        super().__init__()

if __name__ == "__main__":
    tools = Tools()
    # tools.draw_history("accuracy", "/home/mizuno/DeepImFam/result/mizuta/PUNT030102QIAN880126/256/history.csv")
    # tools.draw_history("loss", "/home/mizuno/DeepImFam/result/mizuta/PUNT030102QIAN880126/256/history.csv")
