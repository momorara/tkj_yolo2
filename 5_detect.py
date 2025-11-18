# -*- coding: utf-8 -*-
# -----------------------------------------
# qで終了
# -----------------------------------------
"""
license
GNU Affero General Public License v3（AGPL v3）

yolo detect train data=data.yaml model=yolov8n.pt epochs=3 imgsz=320
として処理した結果を使う

対象画像は.  SOURCE_PATで指定
"""
import os
from ultralytics import YOLO
import shutil
import matplotlib.pyplot as plt
from glob import glob

# --- 設定 ---

# 1. 学習済みモデルのパス
# YOLOv8の学習結果は通常、'runs/detect/train' または 'runs/detect/trainX' に保存されます。
# ここでは、学習が完了した際の最も性能が良い重みファイル (best.pt) を指定します。
MODEL_PATH = 'runs/detect/train/weights/best.pt'


# 2. 検出対象の画像またはフォルダのパス
# 例: 'my_test_image.jpg' または 'path/to/new_images_folder'
SOURCE_PATH = 'catdog.png'

# 3. 結果の保存先ディレクトリ名
PROJECT_NAME = 'custom_inference' 

# 新しい画像を読む前にフォルダを削除
try:
    shutil.rmtree(PROJECT_NAME)
    # print(f"削除しました: {save_dir}")
except Exception as e:
    print(f"ディレクトリ削除失敗: {e}")

# Matplotlib の設定
plt.ion()  # インタラクティブモードON
fig, ax = plt.subplots()
plt.axis('off')

# --- 推論実行 ---
def run_inference():
    """YOLOv8モデルを使って推論を実行する関数"""
    
    # 1. モデルのロード
    # 学習済みモデルの重みファイルを指定してYOLOオブジェクトを作成します。
    try:
        model = YOLO(MODEL_PATH)
        print(f"モデルを正常にロードしました: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません。パスを確認してください: {MODEL_PATH}")
        return

    # 2. 検出の実行
    print(f"推論を {SOURCE_PATH} に対して実行中...")
    
    # predictメソッドを使用して推論を実行
    results = model.predict(
        source=SOURCE_PATH,  # 推論対象
        conf=0.25,           # 信頼度閾値 (デフォルト: 0.25)
        iou=0.7,             # IOU閾値 (重複バウンディングボックスの除去用)
        save=True,           # 検出結果の画像保存を有効化
        project=PROJECT_NAME # 結果を保存するルートディレクトリ名
    )
    jpg_files = glob(os.path.join(results[0].save_dir, "*.jpg"))
    if not jpg_files:
        print("検出結果が見つかりません。")
        return

    img = plt.imread(jpg_files[-1])  # 最新ファイルを読む
    ax.clear()
    ax.imshow(img)
    ax.set_title("YOLO cat or dog")
    ax.axis("off")
    plt.draw()      # 描画を更新

    # 3. 結果の出力場所
    # 結果は通常、プロジェクト名と連番のフォルダに保存されます (例: custom_inference/predict/)
    # 正確な出力パスは実行後にコンソールに表示されます。
    
    print("推論が完了しました。")
    print(f"結果の画像は '{PROJECT_NAME}/predict' のようなフォルダに保存されています。")


if __name__ == "__main__":
    run_inference()
    plt.show(block=True)