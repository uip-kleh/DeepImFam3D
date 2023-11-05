import os
import yaml

class SetConfig:
    yaml_path = "config.yaml"

    def __init__(self) -> None:
        self.new_path = []

        # Yamlファイルを開く
        with open(self.yaml_path) as f:
            args = yaml.safe_load(f)

            # データのパス
            self.gpcr_path = args["gpcr_path"]
            self.trans_path = os.path.join(self.gpcr_path, "trans.txt")
            self.cv0_path = os.path.join(self.gpcr_path, "cv_0")
            self.gpcr_train_path = os.path.join(self.cv0_path, "train.txt")
            self.gpcr_pest_path = os.path.join(self.cv0_path, "test.txt")

            # 画像生成の設定
            self.FIGSIZE = args["FIGSIZE"]
            self.vectorTimes = args["vectorTimes"]
            self.aaindex1 = args["aaindex1"]
            self.aaindex2 = args["aaindex2"]

        self.makeNew_path()

    # 新しくディレクトリを作る
    def makeNew_path(self):
        for path in self.new_path:
            if not os.path.exists(path):
                os.mkdir(path)

if __name__ == '__main__':
    setconfig = SetConfig()

