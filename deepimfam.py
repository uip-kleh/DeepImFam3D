import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from aaindex import aaindex1
import statistics
import numpy as np
from setconfig import SetConfig
from tools import Tools

class DeepImFam(SetConfig, Tools):
    def __init__(self) -> None:
        super().__init__()

    # GPCRを読み込む
    def load_GPCR(self):
        with open(self.gpcrTrainPath) as f:
            for l in f.readlines():
                subsubfam, aaseq = l.split()

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
                (aaindex1_dict[key] - aaindex1_mean) / aaindex1_std * self.times,
                (aaindex2_dict[key] - aaindex2_mean) / aaindex2_std * self.times,
            ])

        # ベクトルの可視化
        fname = os.path.join(self.result_method, "vectors.pdf")
        self.draw_vectors(self.aavector, fname=fname)

if __name__ == '__main__':
    deepimfam = DeepImFam()
    deepimfam.define_aavector()
