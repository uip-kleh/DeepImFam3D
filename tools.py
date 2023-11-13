import os
import matplotlib.pylab as plt
import json

class IOTools:
    def __init__(self) -> None:
        pass

    def save_json(self, obj, fname):
        with open(fname, "w") as f:
            json.dump(obj, f, indent=2)

    def save_dataframe(self, df, fname):
        df.to_csv(fname)


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

    # ファイルを出力
    def save(self, fname):
        plt.savefig(fname, transparent=True)

    # ファイルを閉じる
    def clear(self, fname):
        plt.cla()
        plt.clf()
        plt.close()


class Tools(IOTools, DrawTools):
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    tools = Tools()
