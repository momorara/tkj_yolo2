# -*- coding: utf-8 -*-
"""
license
GNU Affero General Public License v3（AGPL v3）

現在のデータセットは
dataset_s/
├── train/
│   ├── Cat/
│   └── Dog/
└── val/
    ├── Cat/
    └── Dog/
です。
画像ファィル名は　数字.jpg となっています。

これをYolo8用のデータ構造に変換したいです。
その際に画像ファィル名を
Catディレクトリのものは頭にcatを追加
Dogディレクトリのものは頭にdogを追加
して、以下のような構造にして欲しい

dataset_sy
├─ images/           # すべての画像ファイルをここに配置
   ├── train/        # 学習用画像
   │   ├── cat_001.jpg
   │   └── dog_001.jpg
   └── val/          # 検証用画像
       ├── cat_002.jpg
       └── dog_002.jpg
"""
import os
import shutil

# --- 設定 ---
SOURCE_DIR = "dataset_s"  # 現在のデータセットのルートディレクトリ名
TARGET_DIR = "dataset_sy" # 新しいYOLO形式のデータセットのルートディレクトリ名

# --- ディレクトリ構造の定義 ---
SPLITS = ['train', 'val']
CLASSES = ['Cat', 'Dog']

# 新しいYOLO形式の基本ディレクトリを作成
def create_target_structure():
    print(f"ターゲットディレクトリ '{TARGET_DIR}' を作成中...")
    os.makedirs(os.path.join(TARGET_DIR, 'images'), exist_ok=True)
    for split in SPLITS:
        os.makedirs(os.path.join(TARGET_DIR, 'images', split), exist_ok=True)
    print("ディレクトリ構造の作成が完了しました。")

# ファイルの移動とリネームを実行
def rename_and_move_files():
    for split in SPLITS:
        print(f"\n--- {split.upper()} データの処理を開始 ---")
        for class_name in CLASSES:
            source_path = os.path.join(SOURCE_DIR, split, class_name)
            target_path = os.path.join(TARGET_DIR, 'images', split)

            if not os.path.exists(source_path):
                print(f"⚠️ 警告: ソースパス {source_path} が見つかりませんでした。スキップします。")
                continue

            # クラスフォルダ内のファイルを処理
            for filename in os.listdir(source_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')): # 画像ファイルのみを対象
                    # 新しいファイル名を作成: 'cat_' または 'dog_' を追加
                    prefix = class_name.lower() + "_"
                    new_filename = prefix + filename

                    src_file = os.path.join(source_path, filename)
                    dst_file = os.path.join(target_path, new_filename)

                    try:
                        # ファイルの移動とリネームを同時に実行
                        shutil.copy2(src_file, dst_file)
                        # shutil.move(src_file, dst_file)  # 元ファイルを削除する場合はmoveを使用
                        # print(f"  移動: {filename} -> {new_filename}")
                    except Exception as e:
                        print(f"❌ エラー: {src_file} の処理中にエラーが発生しました: {e}")

# --- メイン処理 ---
if __name__ == "__main__":
    create_target_structure()
    rename_and_move_files()
    print("\n✅ データセットの画像変換が完了しました。")
    print(f"新しい画像は '{TARGET_DIR}/images/' にあります。")
    print("次は、YOLO形式のラベル（.txtファイル）を作成してください。")