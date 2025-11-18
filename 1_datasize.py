# -*- coding: utf-8 -*-
"""
license
GNU Affero General Public License v3（AGPL v3）

datasetから
データは指定の数(%)に絞り込みます。
データの選択はランダムとします。

で、dataset_sを作る データ構造は同じ

dataset_s/
├── train/
│   ├── Cat/
│   └── Dog/
└── val/
    ├── Cat/
    └── Dog/
"""
import os
import shutil
import random

# 元のデータセット
src_base = 'dataset'
# 小規模データセット
dst_base = 'dataset_s'

# クラスと分割
classes = ['Cat', 'Dog']
splits  = ['train', 'val']

# 出力フォルダ削除（もし存在すれば） 
if os.path.exists(dst_base):
    shutil.rmtree(dst_base)
    print(f"削除しました: {dst_base}")

# 出力フォルダ作成
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(dst_base, split, cls), exist_ok=True)

# データをランダム抽出してコピー
# fraction = 0.01  # 100分の1
fraction = 0.1  # 10分の1
# fraction = 0.3  # 30分の1
#fraction = 1.0

for split in splits:
    for cls in classes:
        src_dir = os.path.join(src_base, split, cls)
        dst_dir = os.path.join(dst_base, split, cls)
        
        # 画像リスト取得
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(images)
        
        # fractionに従い抽出
        n_select = max(1, int(len(images) * fraction))  # 1枚以上は必ず選ぶ
        selected_images = images[:n_select]
        print(int(len(images) * fraction),n_select)
        
        # コピー
        for fname in selected_images:
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

print("小規模データセット dataset_s の作成完了！")
