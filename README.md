# SAMを動かしてみる
深層学習の生殖医療応用にSAMを導入してみたいお気持ちで動かしてみるためのリポジトリ。
## 1. 環境
- Python 3.8.5
- torch 1.11.0+cu113

## 2. インストール
### Pytorch
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
### SAM
```shell
cd segment-anything
pip install -e .
```

## 3. 学習済みモデルのダウンロード
```shell
mkdir download_model && cd download_model
# vit_h
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# vit_l
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# vit_b
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 4. コードの簡単な説明
### モデルロード
```python
from segment_anything import sam_model_registry
model_type = 'vit_h'
checkpoint = 'path/to/checkpoint'
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to('cuda')
```
`model_type`に`vit_{b,l,h}`を指定し、`checkpoint`にパスを指定することで読み込める。
### プロンプト指定の場合
```python
from segment_anything import SamPredictor
predictor = SamPredictor(sam)
predictor.set_image(img)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[75,75]])
    point_labels=np.array([1]),
    box=np.array([0,0,150,150]),
    multimask_output=True,
    return_logits=False,
)
```
プロンプトを指定する場合は、`SamPredictor()`を使用。
まず、`.set_image()`で入力画像を渡し、`.predict()`でセグメンテーションできる。

- `point_coords` 点プロンプトの座標
- `point_label` 点プロンプトのラベル
- `box` ボックスプロンプトの座標
- `multimask_output` Trueの場合、確信度の高い3つを出力
- `return_logits` Trueの場合、logitを出力

### 自動(プロンプト指定しない)の場合
```python
from segment_anything import SamAutomaticMaskGenerator
predictor = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    box_nms_thresh=0.7,
    min_mask_region_area=0,

)
masks = predictor.generate(img)
```
プロンプトを指定しない(自動の)場合は、`SamAutomaticMaskGenerator()`を使用。
`.generate()`に入力画像を渡すのみ。

- `points_per_side` 画像の1辺のポイント数。合計n×nのポイントが指定される。
- `pred_iou_thresh` モデルの予測マスク品質の閾値
- `stability_score_thresh` モデルの出力したlogitを2値化するための閾値
- `box_nms_thresh` 重複マスクをフィルタリングするためのIoUカットオフ
- `min_mask_region_area` 指定した数値より小さい面積を持つマスクの連結領域と穴を除去

## 5. テスト
サンプルスクリプト
```shell
python test_sam.py \
    --device cuda \
    --mode auto \
    --model_type vit_h \
    --checkpoint download_model/sam_vit_h_4b8939.pth \
    --data_type test_image \
    --sperm_path datasets/groceries.jpg 
```
その他のスクリプトは[scripts](scripts/)参照。

## 結果
![output](fig/output.png)