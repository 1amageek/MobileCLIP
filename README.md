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

モデルは**Hugging Faceから自動ダウンロード**されます。Pythonスクリプトは不要です！

### 自動ダウンロード（推奨）

初回実行時に自動的にモデルをダウンロードします：

```swift
import MobileCLIP
import MLX

let model = MobileCLIP2()

// 初回実行時：Hugging Faceから自動ダウンロード（1.7GB）
// 2回目以降：ローカルキャッシュを使用（高速）
try await model.loadModelFromHuggingFace()

// ウォームアップ（推奨）
try await model.warmupAsync()
```

**ダウンロード先:**
- macOS: `~/Library/Caches/huggingface/hub/`
- iOS: アプリのCachesディレクトリ

**進捗表示付き:**
```swift
try await model.loadModelFromHuggingFace { progress in
    print("Downloaded: \(progress.fractionCompleted * 100)%")
}
```

### 手動セットアップ（開発者向け）

Pythonスクリプトを使用して事前にダウンロードすることも可能：

```bash
pip install torch transformers safetensors
python convert_to_safetensors.py  # MobileCLIP2-S4をダウンロード
```

その後、Bundleから読み込み：
```swift
try model.loadModelFromBundle()
```

## 💻 使い方

### 基本的な使用例

```swift
import MobileCLIP
import MLX

// モデルを初期化（デフォルト: balanced 64MB GPU cache）
let model = MobileCLIP2()

// メモリプロファイルを指定することも可能：
// let model = MobileCLIP2(memoryProfile: .low)    // 20MB - iPhone SE, 古いデバイス
// let model = MobileCLIP2(memoryProfile: .high)   // 128MB - M1/M2 Mac, iPad Pro
// let model = MobileCLIP2(memoryProfile: .custom(megabytes: 100))  // カスタム

// Hugging Faceから自動ダウンロード（推奨）
try await model.loadModelFromHuggingFace()

// ウォームアップ（Metal kernelのコンパイル）
try await model.warmupAsync()

// モデル情報を表示
model.printModelInfo()

// 画像をエンコード
let image = MLXArray.ones([1, 3, 224, 224]) * 0.5  // ダミー画像
let imageEmbedding = try model.encodeImage(image)

print("Image embedding shape: \(imageEmbedding.shape)")
```

### その他のロード方法

```swift
// Bundleから読み込む（事前にモデルファイルを含める場合）
try await model.loadModelFromBundleAsync()

// ローカルパスから読み込む
try await model.loadModelAsync(from: "/path/to/model")
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
