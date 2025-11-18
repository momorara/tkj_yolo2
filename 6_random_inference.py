# -*- coding: utf-8 -*-
# -----------------------------------------
# ランダム画像を選んで推論し、
# スペースで次の画像、qで終了
# -----------------------------------------
"""
license
GNU Affero General Public License v3（AGPL v3）
"""

import os
import random
from glob import glob
import matplotlib.pyplot as plt
from ultralytics import YOLO
import shutil

# モデルとデータセットのパス
MODEL_PATH = 'runs/detect/train/weights/best.pt'
base_dir = 'dataset_s/val'
categories = ['Cat', 'Dog']

# 画像リスト作成
img_list = []
for category in categories:
    folder_path = os.path.join(base_dir, category)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_list.extend(files)

if not img_list:
    print("画像ファイルが見つかりません。")
    exit()

# モデル読み込み
model = YOLO(MODEL_PATH)
print(f"モデルを正常にロードしました: {MODEL_PATH}")

PROJECT_NAME = 'custom_inference'

# Matplotlib の設定
plt.ion()  # インタラクティブモードON
fig, ax = plt.subplots()
plt.axis('off')

def show_random_image():
    """ランダムに画像を選んでYOLOで推論し表示"""
    img_path = random.choice(img_list)
    results = model.predict(
        source=img_path,
        conf=0.25,
        iou=0.7,
        save=True,
        project=PROJECT_NAME,
        exist_ok=True  # フォルダ上書きOK
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
    plt.pause(0.1)  # 少し待つ

def on_key(event):
    if event.key == ' ':
        # 新しい画像を読む前にフォルダを削除
        try:
            shutil.rmtree(PROJECT_NAME)
            # print(f"削除しました: {save_dir}")
        except Exception as e:
            print(f"ディレクトリ削除失敗: {e}")
        show_random_image()
    elif event.key.lower() == 'q':
        plt.close(fig)
        plt.ioff()  # 終了時にオフ

# キーイベント登録
fig.canvas.mpl_connect('key_press_event', on_key)

# 初回表示
show_random_image()

print("スペースキーで次の画像、Qキーで終了。")
plt.show(block=True)