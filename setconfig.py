import os
import yaml
import json

class SetConfig:
    yaml_path = "config.yaml"

    def __init__(self) -> None:
        self.new_path = []
        args = None

        # Yamlファイルを開く
        with open(self.yaml_path) as f:
            args = yaml.safe_load(f)

            # データのパス
            self.gpcr_path = args["gpcr_path"]
            self.trans_path = os.path.join(self.gpcr_path, "trans.txt")
            self.cv0_path = os.path.join(self.gpcr_path, "cv_0")
            self.gpcr_train_path = os.path.join(self.cv0_path, "train.txt")
            self.gpcr_test_path = os.path.join(self.cv0_path, "test.txt")

            # 画像生成の設定
            self.is_gray = args["is_gray"]
            self.method = args["method"]
            self.FIGSIZE = args["FIGSIZE"]
            self.aaindex1 = args["aaindex1"]
            self.aaindex2 = args["aaindex2"]
            self.aaindex3 = args["aaindex3"]

            # モデルの設定
            self.dropout_ratio = args["dropout_ratio"]
            self.max_learning_rate = args["max_learning_rate"]
            self.min_learning_rate = args["min_learning_rate"]
            self.epochs = args["epochs"]
            self.batch_size = args["batch_size"]

        # 生成したデータ
        self.data_method = os.path.join(self.cv0_path, self.method)
        self.new_path.append(self.data_method)
        self.data_aaindex = os.path.join(self.data_method, self.aaindex1 + self.aaindex2 + self.aaindex3)
        self.new_path.append(self.data_aaindex)
        color = "binary"
        if self.is_gray: color = "gray"
        self.color_path = os.path.join(self.data_aaindex, color)
        self.new_path.append(self.color_path)
        self.data_size = os.path.join(self.color_path, str(self.FIGSIZE))
        self.new_path.append(self.data_size)
        self.data_dat = os.path.join(self.data_aaindex, "dat")
        self.new_path.append(self.data_dat)
        self.data_img = os.path.join(self.data_size, "img")
        self.new_path.append(self.data_img)
        self.images_path = os.path.join(self.data_size, "images_path.csv")

        # 結果の保存
        self.result = "result"
        self.new_path.append(self.result)
        self.result_method = os.path.join(self.result, self.method)
        self.new_path.append(self.result_method)
        self.result_aaindex = os.path.join(self.result_method, self.aaindex1 + self.aaindex2 + self.aaindex3)
        self.new_path.append(self.result_aaindex)
        self.result_color = os.path.join(self.result_aaindex, color)
        self.new_path.append(self.result_color)
        self.result_size = os.path.join(self.result_color, str(self.FIGSIZE))
        self.new_path.append(self.result_size)

        self.history_fname = os.path.join(self.result_size, "history.csv")
        self.model_fname = os.path.join(self.result_size, "model.h5")


        # 新規ディレクトリを作成
        self.makeNew_path()

        # configを保存
        fname = os.path.join(self.result_size, "config.json")
        with open(fname, "w") as f:
            json.dump(args, f, indent=1)

    # 新しくディレクトリを作る
    def makeNew_path(self):
        for path in self.new_path:
            if not os.path.exists(path):
                os.mkdir(path)

if __name__ == '__main__':
    setconfig = SetConfig()

