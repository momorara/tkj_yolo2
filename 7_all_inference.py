# -*- coding: utf-8 -*-
"""
license
GNU Affero General Public License v3（AGPL v3）

犬猫分類 YOLO 推論サンプル（全画像順次処理）
- Ultralytics YOLOv8 学習済みモデルを使用
- valフォルダ内の全画像に対して推論し、結果を表示
"""

import os
import shutil
from ultralytics import YOLO

# モデルとデータセットのパス
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# 学習済みYOLOモデル
model = YOLO(MODEL_PATH)  # 学習済みYOLOモデルファイル

# valフォルダのパス
val_dir = 'dataset_s/val'
classes = ['Cat', 'Dog']  # フォルダ名がクラス名
err_dir = 'result_err'

# result_errを削除して新規作成
if os.path.exists(err_dir):
    shutil.rmtree(err_dir)
os.makedirs(err_dir, exist_ok=True)

err_save = 1

dog_n, dogdog, dogcat = 0, 0, 0
cat_n, catcat, catdog = 0, 0, 0

for cls in classes:
    cls_dir = os.path.join(val_dir, cls)
    print(cls, 'start')
    
    for fname in os.listdir(cls_dir):
        if "._" in fname:
            continue
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(cls_dir, fname)
        try:
            # 推論
            results = model.predict(img_path, verbose=False)
            # 最初のフレームの結果を取得
            boxes = results[0].boxes
            
            predicted_class = None
            if len(boxes) > 0:
                # 最も高い信頼度のボックスを使用
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                max_idx = confs.argmax()
                pred_cls_idx = cls_ids[max_idx]
                pred_cls_name = model.names[pred_cls_idx]  # YOLOクラス名
                if pred_cls_name.lower() == 'cat':
                    predicted_class = '猫'
                elif pred_cls_name.lower() == 'dog':
                    predicted_class = '犬'
            else:
                predicted_class = '不明'
            
            # 正解との比較
            if cls == 'Cat':
                cat_n += 1
                if predicted_class == '猫':
                    catcat += 1
                else:
                    catdog += 1
                    if err_save == 1:
                        shutil.copy(img_path, err_dir)
            elif cls == 'Dog':
                dog_n += 1
                if predicted_class == '犬':
                    dogdog += 1
                else:
                    dogcat += 1
                    if err_save == 1:
                        shutil.copy(img_path, err_dir)
        except Exception as e:
            print('エラー:', img_path, e)
    
    print(cls, 'end')

# 結果表示
print('dog:', dog_n, dogdog, dogcat, int(dogdog/dog_n*1000)/10, '%')
print('cat:', cat_n, catcat, catdog, int(catcat/cat_n*1000)/10, '%')
