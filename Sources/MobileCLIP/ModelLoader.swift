import Foundation
import MLX

/// MobileCLIP2モデルの重みをロードするクラス
public class ModelLoader {

    /// ロードされた重み
    public private(set) var weights: [String: MLXArray] = [:]

    public init() {}

    /// Load model weights from safetensors file
    /// - Parameter url: URL to the safetensors file
    /// - Throws: File not found or loading error
    public func loadFromSafetensors(url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.fileNotFound("Model file not found at \(url.path)")
        }

        // MLX.loadArrays() supports safetensors format
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
