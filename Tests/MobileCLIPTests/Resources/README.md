# Test Images

このディレクトリには、テスト用の画像を配置してください。

## 画像の準備方法

### 方法1: サンプル画像をダウンロード

```bash
# 猫の画像
curl -o test_cat.jpg https://placekitten.com/224/224

# 犬の画像
curl -o test_dog.jpg https://placedog.net/224/224

# 汎用テスト画像
cp test_cat.jpg test_image.jpg
```

### 方法2: 自分の画像を使用

1. お好きな画像（JPEG、PNG形式）を用意
2. このディレクトリに `test_image.jpg` という名前でコピー

## テストの実行

画像を配置したら、以下のテストが実行可能になります：

- `encodeRealImage()` - 実際の画像をエンコード
- `realImageTextSimilarity()` - 画像とテキストの類似度計算
- `zeroShotClassificationRealImage()` - ゼロショット分類

## 注意事項

- 画像は自動的に224x224にリサイズされます
- ImageNet統計で正規化されます（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
