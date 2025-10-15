import Testing
import Foundation
import MLX
import MLXLMCommon
import CoreGraphics
import AppKit
@testable import MobileCLIP

// MARK: - Real Image Tests

@Suite("Real Image Tests")
struct RealImageTests {

    // MARK: - Helper Functions

    /// 画像ファイルをMLXArrayに変換
    /// - Parameter imagePath: 画像ファイルのパス
    /// - Returns: MLXArray in NCHW format [1, 3, 224, 224]
    static func loadImage(from imagePath: String) throws -> MLXArray {
        guard let nsImage = NSImage(contentsOfFile: imagePath) else {
            throw NSError(domain: "RealImageTests", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load image from \(imagePath)"])
        }

        // NSImageをCGImageに変換
        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "RealImageTests", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to convert NSImage to CGImage"])
        }

        // 224x224にリサイズ
        let targetSize = CGSize(width: 224, height: 224)
        guard let resizedImage = resizeImage(cgImage, to: targetSize) else {
            throw NSError(domain: "RealImageTests", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
        }

        // CGImageからピクセルデータを抽出
        let width = 224
        let height = 224
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8

        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )

        context?.draw(resizedImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // [0-255] UInt8 → [0-1] Float に正規化し、ImageNet統計で標準化
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        var imageArray = [Float](repeating: 0, count: 3 * height * width)

        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * width + x
                let dataIndex = pixelIndex * bytesPerPixel

                let r = Float(pixelData[dataIndex]) / 255.0
                let g = Float(pixelData[dataIndex + 1]) / 255.0
                let b = Float(pixelData[dataIndex + 2]) / 255.0

                // NCHW format: [channels, height, width]
                imageArray[0 * height * width + pixelIndex] = (r - mean[0]) / std[0]
                imageArray[1 * height * width + pixelIndex] = (g - mean[1]) / std[1]
                imageArray[2 * height * width + pixelIndex] = (b - mean[2]) / std[2]
            }
        }

        // MLXArray作成: [1, 3, 224, 224]
        let mlxArray = MLXArray(imageArray, [1, 3, height, width])
        return mlxArray
    }

    /// CGImageをリサイズ
    private static func resizeImage(_ image: CGImage, to size: CGSize) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 4 * width,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return nil
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()
    }

    // MARK: - Helper to get test image path

    /// Bundleからテスト画像のパスを取得
    static func getTestImagePath(filename: String = "test_image.jpg") -> String? {
        // Bundle.moduleのリソースURLを取得
        guard let bundleResourceURL = Bundle.module.resourceURL else {
            return nil
        }

        // まず直接確認
        var url = bundleResourceURL.appendingPathComponent(filename)

        // 見つからない場合は Resources/ サブディレクトリを試す
        if !FileManager.default.fileExists(atPath: url.path) {
            url = bundleResourceURL.appendingPathComponent("Resources").appendingPathComponent(filename)
        }

        // 存在を確認
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }

        return url.path
    }

    // MARK: - Tests with Real Images

    @Test("Encode real image from file")
    func encodeRealImage() async throws {
        // Bundleからテスト画像のパスを取得
        guard let testImagePath = Self.getTestImagePath() else {
            print("⚠️ Test image not found in bundle")
            print("   Please add 'test_image.jpg' to Tests/MobileCLIP2Tests/Resources/")
            return  // テスト画像がない場合はスキップ
        }

        let model = MobileCLIP()
        try await model.load(configuration: ModelConfiguration(id: "1amageek/MobileCLIP2-S4"))

        let imageArray = try Self.loadImage(from: testImagePath)
        print("✅ Loaded image: \(imageArray.shape)")

        let embedding = try model.encodeImage(imageArray)
        print("✅ Image embedding: \(embedding.shape)")

        #expect(embedding.shape[0] == 1, "Batch size should be 1")
        #expect(embedding.shape[1] == 768, "Embedding dimension should be 768")

        // L2正規化の確認
        let norm = embedding.square().sum(axis: -1).sqrt()
        let normValue = norm.item(Float.self)
        #expect(abs(normValue - 1.0) < 0.01, "Embedding should be L2 normalized")

        print("✅ Embedding norm: \(normValue)")
    }

    @Test("Real image-text similarity")
    func realImageTextSimilarity() async throws {
        // Bundleからテスト画像のパスを取得
        guard let testImagePath = Self.getTestImagePath() else {
            print("⚠️ Test image not found in bundle")
            print("   Please add 'test_image.jpg' to Tests/MobileCLIP2Tests/Resources/")
            return
        }

        let model = MobileCLIP()
        try await model.load(configuration: ModelConfiguration(id: "1amageek/MobileCLIP2-S4"))

        // 画像をエンコード
        let imageArray = try Self.loadImage(from: testImagePath)
        let imageEmbedding = try model.encodeImage(imageArray)

        // 複数のテキストとの類似度を計算
        let texts = [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a bird",
            "a photo of a car",
            "a photo of a flower"
        ]

        let tokenizer = try Tokenizer()
        var similarities = [Float]()

        for text in texts {
            let tokens = tokenizer.encode(text)
            let textEmbedding = try model.encodeText(tokens)
            let similarity = model.computeSimilarity(
                imageEmbedding: imageEmbedding,
                textEmbedding: textEmbedding
            )
            similarities.append(similarity)
        }

        print("✅ Image-Text Similarities:")
        for (i, text) in texts.enumerated() {
            print("   \(text): \(similarities[i])")
        }

        // すべての類似度が有効な範囲内にあることを確認
        for similarity in similarities {
            #expect(similarity >= -1.0 && similarity <= 1.0, "Similarity should be in [-1, 1]")
            #expect(!similarity.isNaN, "Similarity should not be NaN")
        }
    }

    @Test("Zero-shot classification with real image")
    func zeroShotClassificationRealImage() async throws {
        // Bundleからテスト画像のパスを取得
        guard let testImagePath = Self.getTestImagePath() else {
            print("⚠️ Test image not found in bundle")
            print("   Please add 'test_image.jpg' to Tests/MobileCLIP2Tests/Resources/")
            return
        }

        let model = MobileCLIP()
        try await model.load(configuration: ModelConfiguration(id: "1amageek/MobileCLIP2-S4"))

        // 画像をエンコード
        let imageArray = try Self.loadImage(from: testImagePath)
        let imageEmbedding = try model.encodeImage(imageArray)

        // 分類用のラベル
        let labels = [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a bird"
        ]

        let tokenizer = try Tokenizer()
        var logits = [Float]()

        // 各ラベルとの類似度を計算
        for label in labels {
            let tokens = tokenizer.encode(label)
            let textEmbedding = try model.encodeText(tokens)
            let similarity = model.computeSimilarity(
                imageEmbedding: imageEmbedding,
                textEmbedding: textEmbedding
            )
            logits.append(similarity)
        }

        // Softmax適用
        let maxLogit = logits.max() ?? 0
        let expLogits = logits.map { exp($0 - maxLogit) }
        let sumExp = expLogits.reduce(0, +)
        let probabilities = expLogits.map { $0 / sumExp }

        print("✅ Zero-shot Classification:")
        for (i, label) in labels.enumerated() {
            print("   \(label): \(probabilities[i] * 100)%")
        }

        // 最も確率が高いラベルを取得
        if let maxIndex = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) {
            print("   Predicted: \(labels[maxIndex]) (\(probabilities[maxIndex] * 100)%)")
        }

        // 確率の合計が1に近いことを確認
        let sumProb = probabilities.reduce(0, +)
        #expect(abs(sumProb - 1.0) < 0.01, "Probabilities should sum to 1.0")
    }
}

// MARK: - Download Test Image Helper

@Suite("Test Image Setup")
struct TestImageSetup {

    @Test("Instructions for adding test images", .disabled("Information only"))
    func testImageInstructions() {
        print("""

        📸 テスト画像の準備方法:

        1. テスト用の画像を用意してください（JPEG, PNG形式）

        2. 以下のディレクトリに画像を配置してください:
           /Users/1amageek/Desktop/CLIP/MobileCLIP2App/Tests/MobileCLIP2Tests/Resources/

        3. ファイル名を以下のいずれかにしてください:
           - test_image.jpg
           - test_cat.jpg (猫の画像)
           - test_dog.jpg (犬の画像)

        4. Xcodeで Package.swift を更新して、リソースを含めるようにしてください:

           .testTarget(
               name: "MobileCLIP2Tests",
               dependencies: ["MobileCLIP2"],
               resources: [.copy("Resources")]
           )

        5. テストを実行してください

        または、コマンドラインから画像をダウンロード:

        # Resourcesディレクトリを作成
        mkdir -p Tests/MobileCLIP2Tests/Resources

        # サンプル画像をダウンロード（例）
        curl -o Tests/MobileCLIP2Tests/Resources/test_cat.jpg \\
          https://placekitten.com/224/224

        """)
    }
}
