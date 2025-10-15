import Foundation
import MLX

/// MobileCLIP2モデルの重みをロードするクラス
public class ModelLoader {

    /// ロードされた重み
    public private(set) var weights: [String: MLXArray] = [:]

    public init() {}

    /// 4つに分割された.npzファイルをすべて読み込む
    /// - Parameter basePath: モデルファイルのベースパス（拡張子なし）
    /// - Throws: ファイルが見つからない、または読み込みエラー
    public func loadWeights(basePath: String) throws {
        for partNumber in 1...4 {
            let filename = "\(basePath)_part\(String(format: "%02d", partNumber)).npz"
            let url = URL(fileURLWithPath: filename)

            guard FileManager.default.fileExists(atPath: filename) else {
                throw ModelError.fileNotFound(filename)
            }

            // .npz ファイルを読み込む
            let partWeights = try MLX.loadArrays(url: url)

            // 既存の weights に追加
            for (key, value) in partWeights {
                weights[key] = value
            }
        }
    }

    /// Bundleからsafetensorsファイルをロード
    /// - Parameter resourceName: リソース名（拡張子なし）
    /// - Throws: ファイルが見つからない、または読み込みエラー
    public func loadFromBundle(resourceName: String = "MobileCLIP2-S4") throws {
        // Bundle.moduleのリソースURLを取得
        guard let bundleResourceURL = Bundle.module.resourceURL else {
            throw ModelError.fileNotFound("Bundle.module.resourceURL not found")
        }

        // safetensorsファイルを読み込む
        // .copy("Resources")を使うと、Bundle構造が異なる:
        // - swift test: .bundle/Resources/MobileCLIP2-S4.safetensors
        // - Xcode: .bundle/Contents/Resources/Resources/MobileCLIP2-S4.safetensors
        let safetensorsFile = "\(resourceName).safetensors"

        // まず直接確認
        var url = bundleResourceURL.appendingPathComponent(safetensorsFile)

        // 見つからない場合は Resources/ サブディレクトリを試す
        if !FileManager.default.fileExists(atPath: url.path) {
            url = bundleResourceURL.appendingPathComponent("Resources").appendingPathComponent(safetensorsFile)
        }

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.fileNotFound("\(safetensorsFile) not found at \(url.path)")
        }

        // MLX.loadArrays()はsafetensors形式をサポート
        weights = try MLX.loadArrays(url: url)
    }

    /// 特定のキーの重みを取得
    /// - Parameter key: 重みのキー
    /// - Returns: MLXArray、存在しない場合は nil
    public func getWeight(_ key: String) -> MLXArray? {
        return weights[key]
    }

    /// すべての重みのキーを取得
    public var allKeys: [String] {
        return Array(weights.keys).sorted()
    }

    /// 特定のプレフィックスで始まる重みをフィルタ
    /// - Parameter prefix: プレフィックス
    /// - Returns: フィルタされた重み
    public func filterWeights(prefix: String) -> [String: MLXArray] {
        return weights.filter { $0.key.hasPrefix(prefix) }
    }
}

/// モデル関連のエラー
public enum ModelError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidModelStructure
    case inferenceError(String)
    case weightNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Model file not found: \(path)"
        case .invalidModelStructure:
            return "Invalid model structure"
        case .inferenceError(let message):
            return "Inference error: \(message)"
        case .weightNotFound(let key):
            return "Weight not found: \(key)"
        }
    }
}
