# MobileCLIP2 Swift Package

MLX Swiftを使用したMobileCLIP2-S4の実装

## 📋 概要

このパッケージは、Apple MobileCLIP2-S4モデルをMLX Swiftで実装したものです。
iOS 17+およびmacOS 14+で動作します。

## 🚀 インストール

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/MobileCLIP.git", from: "1.0.0")
]
```

## 📦 モデルのセットアップ

モデルファイルは容量が大きいため、Hugging Faceから自動ダウンロードする仕組みになっています。

### 1. 依存パッケージのインストール

```bash
pip install torch transformers safetensors
```

### 2. モデルのダウンロードと変換

```bash
# MobileCLIP2-S4をダウンロード（デフォルト）
python convert_to_safetensors.py

# または他のモデルを指定
python convert_to_safetensors.py --model apple/MobileCLIP2-S2
```

このスクリプトは以下を実行します：
1. Hugging Faceから指定モデルをダウンロード
2. safetensors形式に変換
3. `Sources/MobileCLIP/Resources/MobileCLIP2-S4.safetensors` に保存

### 利用可能なモデル

- `apple/MobileCLIP2-S0` (11.4M params, 71.5% ImageNet)
- `apple/MobileCLIP2-S2` (35.7M params, 77.2% ImageNet)
- `apple/MobileCLIP2-B` (86.3M params, 79.4% ImageNet)
- `apple/MobileCLIP2-S3` (125.1M params, 80.7% ImageNet)
- `apple/MobileCLIP2-L-14` (304.3M params, 81.9% ImageNet)
- `apple/MobileCLIP2-S4` (321.6M params, 81.9% ImageNet) **[デフォルト]**

## 💻 使い方

### 基本的な使用例

```swift
import MobileCLIP
import MLX

// Initialize model with default settings (balanced: 64MB GPU cache)
let model = MobileCLIP2()

// Or specify memory profile for your device:
// let model = MobileCLIP2(memoryProfile: .low)    // 20MB - iPhone SE, older devices
// let model = MobileCLIP2(memoryProfile: .high)   // 128MB - M1/M2 Macs, iPad Pro
// let model = MobileCLIP2(memoryProfile: .custom(megabytes: 100))  // Custom

// Load model from Bundle (recommended)
try model.loadModelFromBundle()

// Or load from local path
// try model.loadModel(from: "/path/to/MobileCLIP2-S4")

// Optional: Warm up the model to compile Metal kernels
try model.warmup()

// Print model info
model.printModelInfo()

// Encode image
let image = MLXArray.ones([1, 3, 224, 224]) * 0.5  // Dummy image
let imageEmbedding = try model.encodeImage(image)

print("Image embedding shape: \(imageEmbedding.shape)")
```

### 画像の前処理

```swift
import UIKit

// UIImageをMLXArrayに変換（実装例）
func preprocessImage(_ image: UIImage) -> MLXArray {
    // 1. 224x224にリサイズ
    let resized = image.resized(to: CGSize(width: 224, height: 224))

    // 2. ピクセルデータを取得
    // 3. [0, 255] → [0, 1] に正規化
    // 4. ImageNet統計で正規化
    // 5. [H, W, C] → [C, H, W] に変換

    // TODO: 実装
    return MLXArray.zeros([1, 3, 224, 224])
}
```

## 🧪 テスト

```bash
swift test
```

## 📚 構成

### ファイル構造

```
MobileCLIP2App/
├── Package.swift
├── README.md
├── Sources/
│   └── MobileCLIP2/
│       ├── MobileCLIP2.swift        # メインモデルクラス
│       ├── ModelLoader.swift        # モデルローダー
│       ├── VisionEncoder.swift      # Vision Encoder
│       ├── Layers.swift             # 基本レイヤー
│       └── Resources/               # .npzファイル（追加必要）
└── Tests/
    └── MobileCLIP2Tests/
        └── MobileCLIP2Tests.swift
```

### 主要クラス

#### `MobileCLIP2`
- メインモデルクラス
- 画像・テキストのエンコード
- 類似度計算

#### `ModelLoader`
- `.npz`ファイルの読み込み
- 重みの管理

#### `VisionEncoder`
- 画像エンコーダ
- Stem, Stages, Head の実装

#### `Layers`
- Conv2d, BatchNorm2d, ReLU等
- 基本的なニューラルネットワークレイヤー

## ⚠️ 制限事項

### 現在の実装状態

- ✅ モデルローダー（完成）
- ✅ 基本レイヤー（完成）
- ⚠️ Vision Encoder（部分実装）
  - ✅ Stem layers
  - ⚠️ Stage blocks（簡易実装）
  - ✅ Head
- ❌ Text Encoder（未実装）
- ❌ 画像前処理（未実装）
- ❌ テキストトークナイザ（未実装）

### 完全実装に必要な作業

1. **Vision Encoder の完成**
   - Token Mixer の実装
   - Channel Mixer の実装
   - 正確なブロック数の設定

2. **Text Encoder の実装**
   - Transformer layers
   - Attention mechanism
   - MLP layers

3. **前処理パイプライン**
   - 画像のリサイズと正規化
   - テキストのトークン化

## 🔧 開発環境

- Xcode 15.0+
- Swift 6.0+
- macOS 14.0+ / iOS 17.0+
- MLX Swift (latest)

## 📖 参考

- [MobileCLIP論文](https://machinelearning.apple.com/research/mobileclip)
- [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip)
- [apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

プルリクエストを歓迎します！

## 👤 作者

[@1amageek](https://github.com/1amageek)

## 📝 更新履歴

- **v1.0.0** (2025-10-14)
  - 初期リリース
  - モデルローダー実装
  - Vision Encoder 部分実装
