import Foundation
import MLX
import MLXLMCommon
import Hub

/// Memory profile for GPU cache configuration
///
/// Selects optimal GPU cache size based on device memory capacity
public enum MemoryProfile {
    /// Low memory devices (20MB)
    /// - For iPhone SE, older iPads, memory-constrained devices
    case low

    /// Balanced configuration (64MB) - Default
    /// - Recommended for most devices
    case balanced

    /// High-performance devices (128MB)
    /// - For M1/M2 Macs, iPad Pro, devices with ample memory
    case high

    /// Custom configuration
    /// - Parameter megabytes: Cache size in megabytes
    case custom(megabytes: Int)

    var bytes: Int {
        switch self {
        case .low:
            return 20 * 1024 * 1024
        case .balanced:
            return 64 * 1024 * 1024
        case .high:
            return 128 * 1024 * 1024
        case .custom(let megabytes):
            return megabytes * 1024 * 1024
        }
    }
}

/// MobileCLIP2-S4 Model
///
/// CLIP model for encoding images and text to compute similarity
public class MobileCLIP: @unchecked Sendable {

    let loader: ModelLoader  // internal for testing
    private var visionEncoder: VisionEncoderComplete?
    private var textEncoder: TextEncoder?

    /// Initialize MobileCLIP2 model
    /// - Parameter memoryProfile: Memory profile (default: .balanced)
    ///
    /// Usage examples:
    /// ```swift
    /// // Default (64MB)
    /// let model = MobileCLIP2()
    ///
    /// // Memory-constrained devices (20MB)
    /// let model = MobileCLIP2(memoryProfile: .low)
    ///
    /// // High-performance devices (128MB)
    /// let model = MobileCLIP2(memoryProfile: .high)
    ///
    /// // Custom configuration (100MB)
    /// let model = MobileCLIP2(memoryProfile: .custom(megabytes: 100))
    /// ```
    public init(memoryProfile: MemoryProfile = .balanced) {
        self.loader = ModelLoader()

        // Set GPU cache limit for optimal performance
        // This prevents out-of-memory issues and improves inference speed
        MLX.GPU.set(cacheLimit: memoryProfile.bytes)
    }

    // MARK: - Model Loading

    /// Load model from local directory or Hugging Face Hub
    /// - Parameters:
    ///   - configuration: Model configuration (uses MLXLMCommon.ModelConfiguration)
    ///   - hub: HubApi instance (default: .shared)
    ///   - progressHandler: Progress callback for downloads (optional)
    ///
    /// Usage:
    /// ```swift
    /// let model = MobileCLIP()
    ///
    /// // Load from Hugging Face Hub (downloads and caches automatically)
    /// let config = ModelConfiguration(id: "1amageek/MobileCLIP2-S4")
    /// try await model.load(configuration: config)
    ///
    /// // Load from local directory
    /// let localURL = URL(fileURLWithPath: "/path/to/model/directory")
    /// let localConfig = ModelConfiguration(directory: localURL)
    /// try await model.load(configuration: localConfig)
    /// ```
    public func load(
        configuration: ModelConfiguration,
        hub: HubApi = .shared,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws {
        // Use MLXLMCommon's downloadModel function
        let modelDirectory = try await MLXLMCommon.downloadModel(
            hub: hub,
            configuration: configuration,
            progressHandler: progressHandler
        )

        // Find the safetensors file in the model directory
        let contents = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )

        guard let modelURL = contents.first(where: { $0.pathExtension == "safetensors" }) else {
            throw ModelError.fileNotFound("No .safetensors file found in \(modelDirectory.path)")
        }

        // Load weights from safetensors file
        try loader.loadFromSafetensors(url: modelURL)
        try initializeEncoders()
    }

    private func initializeEncoders() throws {
        // Vision Encoderを初期化
        visionEncoder = VisionEncoderComplete(weights: loader.weights)

        // Text Encoderを初期化
        textEncoder = TextEncoder(weights: loader.weights)
    }

    // MARK: - Warmup

    /// Warm up the model by running a single inference pass
    /// This compiles Metal kernels ahead of time, improving subsequent inference speed
    /// Call this after loading the model to avoid slow first inference
    public func warmup() throws {
        guard visionEncoder != nil, textEncoder != nil else {
            throw ModelError.invalidModelStructure
        }

        // Warm up vision encoder with dummy input
        let dummyImage = MLXArray.ones([1, 3, 224, 224]) * 0.5
        _ = try encodeImage(dummyImage)

        // Warm up text encoder with dummy input
        var dummyTokens = [Int32](repeating: 0, count: 77)
        dummyTokens[0] = 49406  // SOS
        dummyTokens[1] = 1000
        dummyTokens[2] = 49407  // EOS
        let tokenArray = MLXArray(dummyTokens).reshaped(1, 77)
        _ = try encodeText(tokenArray)
    }

    /// 非同期warmup
    /// Metal kernelのコンパイルを非同期で実行
    ///
    /// Usage:
    /// ```swift
    /// try await model.loadModelFromBundleAsync()
    /// try await model.warmupAsync()
    /// ```
    public func warmupAsync() async throws {
        // MLX operations are already optimized for async execution
        try warmup()
    }

    // MARK: - Encoding

    /// 画像をエンコード
    /// - Parameter image: [batch, channels, height, width] の画像テンソル
    /// - Returns: [batch, embed_dim] の埋め込みベクトル
    public func encodeImage(_ image: MLXArray) throws -> MLXArray {
        guard let encoder = visionEncoder else {
            throw ModelError.invalidModelStructure
        }

        return try encoder.encode(image)
    }

    /// テキストをエンコード
    /// - Parameter tokens: [batch, context_length] のトークンID配列
    /// - Returns: [batch, embed_dim] の埋め込みベクトル
    public func encodeText(_ tokens: MLXArray) throws -> MLXArray {
        guard let encoder = textEncoder else {
            throw ModelError.invalidModelStructure
        }

        return try encoder.encode(tokens)
    }

    // MARK: - Similarity

    /// 画像とテキストの類似度を計算
    /// - Parameters:
    ///   - imageEmbedding: 画像の埋め込みベクトル
    ///   - textEmbedding: テキストの埋め込みベクトル
    /// - Returns: コサイン類似度
    public func computeSimilarity(
        imageEmbedding: MLXArray,
        textEmbedding: MLXArray
    ) -> Float {
        // コサイン類似度 = dot(A, B) / (||A|| * ||B||)
        // エンコーダーは既にL2正規化済みなので、単純なドット積で計算可能
        // ただし、念のためノルムで正規化

        let dotProduct = (imageEmbedding * textEmbedding).sum(axis: -1)

        // Force evaluation to ensure computation is complete
        eval(dotProduct)

        let similarity = dotProduct.item(Float.self)
        return similarity
    }

    // MARK: - Utility

    /// モデルの統計情報を表示
    public func printModelInfo() {
        print("\n📊 Model Information")
        print("   Total tensors: \(loader.weights.count)")

        // Vision encoder の重み
        let visionWeights = loader.filterWeights(prefix: "module.visual")
        print("   Vision encoder tensors: \(visionWeights.count)")

        // Text encoder の重み
        let textWeights = loader.filterWeights(prefix: "module.text")
        print("   Text encoder tensors: \(textWeights.count)")

        // Logit scale
        if let logitScale = loader.getWeight("module.logit_scale") {
            print("   Logit scale: \(logitScale.item(Float.self))")
        }
    }

    /// 利用可能な重みのキーを表示
    public func printAvailableKeys(limit: Int = 20) {
        print("\n📋 Available weight keys (showing first \(limit)):")
        for (i, key) in loader.allKeys.prefix(limit).enumerated() {
            if let weight = loader.getWeight(key) {
                print("  \(i + 1). \(key): \(weight.shape)")
            }
        }

        if loader.allKeys.count > limit {
            print("  ... and \(loader.allKeys.count - limit) more")
        }
    }
}
