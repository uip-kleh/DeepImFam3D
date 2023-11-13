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
            self.method = args["method"]
            self.FIGSIZE = args["FIGSIZE"]
            self.aaindex1 = args["aaindex1"]
            self.aaindex2 = args["aaindex2"]

        # 生成したデータ
        self.data_method = os.path.join(self.cv0_path, self.method)
        self.new_path.append(self.data_method)
        self.data_aaindex = os.path.join(self.data_method, self.aaindex1 + self.aaindex2)
        self.new_path.append(self.data_aaindex)
        self.data_dat = os.path.join(self.data_aaindex, "dat")
        self.new_path.append(self.data_dat)
        self.data_img = os.path.join(self.data_aaindex, "img")
        self.new_path.append(self.data_img)
        self.images_path = os.path.join(self.data_method, "images_path.csv")

        # 結果の保存
        self.result = "result"
        self.new_path.append(self.result)
        self.result_method = os.path.join(self.result, self.method)
        self.new_path.append(self.result_method)
        self.result_aaindex = os.path.join(self.result_method, self.aaindex1 + self.aaindex2)
        self.new_path.append(self.result_aaindex)

        # 新規ディレクトリを作成
        self.makeNew_path()

        # configを保存
        fname = os.path.join(self.result_aaindex, "config.json")
        with open(fname, "w") as f:
            json.dump(args, f, indent=1)

    # 新しくディレクトリを作る
    def makeNew_path(self):
        for path in self.new_path:
            if not os.path.exists(path):
                os.mkdir(path)

if __name__ == '__main__':
    setconfig = SetConfig()

