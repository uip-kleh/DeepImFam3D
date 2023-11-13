import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from aaindex import aaindex1
import statistics
import numpy as np
import pandas as pd
from setconfig import SetConfig
from tools import Tools

class DeepImFam(SetConfig, Tools):
    def __init__(self) -> None:
        super().__init__()

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
        fname = os.path.join(self.result_method, "vectors.pdf")
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

if __name__ == '__main__':
    deepimfam = DeepImFam()

    # ベクトルを割り当てる
    # deepimfam.define_aavector()

    # 座標を計算する
    # deepimfam.calc_coordinate()

    # 学習を行う
    deepimfam.make_image_info()
