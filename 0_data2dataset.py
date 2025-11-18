# -*- coding: utf-8 -*-
"""
license
GNU Affero General Public License v3（AGPL v3）

https://www.kaggle.com/datasets/chetankv/dogs-cats-images?resource=download
dog vs cat
のデータセットに対応したツール

らライセンスが　CC0 1.0 Universal　なので安心


解凍した元データを分割してdatasetとする
分類タスクで使いやすいように フォルダに分ける：

デレクトリdog vs catのデータを
新しいディレクトリdataset1にコピーする

dog vs cat/
└──dataset/
    └──training_set/
    │   ├──cats/
    │   │  └──cat_1.jpg
    │   └──Dogs/
    │      └──dog_1.jpg
    │
    └──test_set/
        ├──cats/
        │  └──cat_1.jpg
        └──Dogs/
           └──dog_1.jpg

を次の形に変換する
ファィル名は数字のみとする

dataset/
├── train/
│   ├── Cat/
│   └── Dog/
└── val/
    ├── Cat/
    └── Dog/


"""
import os
import shutil

# --- 設定（必要に応じて書き換える） ---
src_base = "dog vs cat/dataset"     # 元データのルート
dst_base = "dataset"               # 出力先ルート

# 元サブフォルダ名（元データの実フォルダ名に合わせる）
src_train = os.path.join(src_base, "training_set")
src_test  = os.path.join(src_base, "test_set")

# 出力のサブフォルダ
dst_train = os.path.join(dst_base, "train")
dst_val   = os.path.join(dst_base, "val")

# クラス名マップ（元フォルダ名 -> 出力フォルダ名）
class_map = {
    "cats": "Cat",
    "dogs": "Dog",
    # 必要ならここに追加
}

# コピー（または移動）モード：'copy' または 'move'
MODE = "copy"

# --- dataset1 ディレクトリを削除して再作成 ---
if os.path.exists(dst_base):
    print(f"🧹 既存のディレクトリを削除します: {dst_base}")
    shutil.rmtree(dst_base)
os.makedirs(dst_base, exist_ok=True)
print(f"✅ 新しいディレクトリを作成しました: {dst_base}")
# -------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def collect_files(folder):
    """指定フォルダ内のファイル一覧（拡張子保持）をソートして返す"""
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    files.sort()  # 名前順で安定化（必要なら別キーでソート）
    return files

def process_split(src_root, dst_root):
    """
    src_root の各クラスフォルダを見て、
    dst_root/<MappedClass>/ に連番でファイルをコピー（または移動）する
    """
    ensure_dir(dst_root)
    for src_class_name, dst_class_name in class_map.items():
        src_dir = os.path.join(src_root, src_class_name)
        dst_dir = os.path.join(dst_root, dst_class_name)
        ensure_dir(dst_dir)

        files = collect_files(src_dir)
        if not files:
            print(f"⚠️ 元フォルダが空または存在しません: {src_dir}")
            continue

        print(f"\n処理: {src_dir} -> {dst_dir} （{len(files)} 件）")
        for idx, fname in enumerate(files, start=1):
            src_path = os.path.join(src_dir, fname)
            # 拡張子を保持して連番ファイル名を作る
            _, ext = os.path.splitext(fname)
            new_name = f"{idx}{ext.lower()}"  # 小文字拡張子に統一
            dst_path = os.path.join(dst_dir, new_name)

            # 既に同名ファイルが存在する場合は、被らないように suffix を付ける（念のため）
            if os.path.exists(dst_path):
                k = 1
                while True:
                    new_name_k = f"{idx}_{k}{ext.lower()}"
                    dst_path = os.path.join(dst_dir, new_name_k)
                    if not os.path.exists(dst_path):
                        new_name = new_name_k
                        break
                    k += 1

            if MODE == "copy":
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

            print(f"  {fname} -> {new_name}")

# --- 実行 ---
if __name__ == "__main__":
    # train
    process_split(src_train, dst_train)
    # val / test
    process_split(src_test, dst_val)

    print("\n完了しました。出力先:", os.path.abspath(dst_base))
    print("MODE =", MODE)
