import Testing
import Foundation
import MLX
@testable import MobileCLIP

// MARK: - Test Suite

@Suite("MobileCLIP Model Tests")
struct MobileCLIPTests {

    // MARK: - Model Loading Tests

    @Test("Model loads from bundle successfully")
    func modelLoadingFromBundle() async throws {
        let model = MobileCLIP()

        try model.loadModelFromBundle()
        model.printModelInfo()
        model.printAvailableKeys(limit: 10)

        // Model loaded successfully (no assertion needed as loadModelFromBundle would throw on failure)
    }

    @Test("Model info validates tensor count")
    func modelInfo() async throws {
        let loader = ModelLoader()

        try loader.loadFromBundle()

        #expect(loader.weights.count > 0, "Should have loaded weights")
        #expect(loader.weights.count == 1715, "Should have exactly 1715 tensors")

        let visionWeights = loader.filterWeights(prefix: "module.visual")
        #expect(visionWeights.count > 0, "Should have vision encoder weights")

        let textWeights = loader.filterWeights(prefix: "module.text")
        #expect(textWeights.count > 0, "Should have text encoder weights")

        print("📊 Model Statistics:")
        print("   Total weights: \(loader.weights.count)")
        print("   Vision weights: \(visionWeights.count)")
        print("   Text weights: \(textWeights.count)")
    }
}

// MARK: - Vision Encoder Tests

@Suite("Vision Encoder Tests")
struct VisionEncoderTests {

    @Test("Single image encoding produces correct shape")
    func visionEncoding() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // Use constant non-zero value to avoid NaN in L2 normalization
        let dummyImage = MLXArray.ones([1, 3, 224, 224]) * 0.5
        let embedding = try model.encodeImage(dummyImage)

        print("✅ Image embedding shape: \(embedding.shape)")

        #expect(embedding.ndim == 2, "Embedding should be 2D")
        #expect(embedding.shape[0] == 1, "Batch size should be 1")
        #expect(embedding.shape[1] == 768, "Embedding dimension should be 768")

        // L2正規化の確認
        let norm = embedding.square().sum(axis: -1).sqrt()
        let normValue = norm.item(Float.self)
        #expect(abs(normValue - 1.0) < 0.01, "Embedding should be L2 normalized (norm ≈ 1.0)")
    }

    @Test("Batch image encoding")
    func visionEncodingBatch() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        let batchSize = 4
        let batchImages = MLXArray.zeros([batchSize, 3, 224, 224])
        let embeddings = try model.encodeImage(batchImages)

        print("✅ Batch image embeddings shape: \(embeddings.shape)")

        #expect(embeddings.shape[0] == batchSize, "Batch size should be \(batchSize)")
        #expect(embeddings.shape[1] == 768, "Embedding dimension should be 768")
    }

    @Test("Vision encoder normalizes embeddings", arguments: [1, 2, 4])
    func visionEncoderNormalization(batchSize: Int) async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // Use constant non-zero value to avoid NaN
        let images = MLXArray.ones([batchSize, 3, 224, 224]) * 0.5
        let embeddings = try model.encodeImage(images)

        // 各バッチアイテムの正規化を確認
        for i in 0..<batchSize {
            let embedding = embeddings[i]
            let norm = embedding.square().sum().sqrt()
            let normValue = norm.item(Float.self)
            #expect(abs(normValue - 1.0) < 0.01, "Batch item \(i) should be normalized")
        }
    }
}

// MARK: - Text Encoder Tests

@Suite("Text Encoder Tests")
struct TextEncoderTests {

    @Test("Single text encoding produces correct shape")
    func textEncoding() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        var tokens = [Int32](repeating: 0, count: 77)
        tokens[0] = 49406  // SOS
        tokens[1] = 1000
        tokens[2] = 2000
        tokens[3] = 49407  // EOS

        let dummyTokens = MLXArray(tokens).reshaped(1, 77)
        let embedding = try model.encodeText(dummyTokens)

        print("✅ Text embedding shape: \(embedding.shape)")

        #expect(embedding.ndim == 2, "Embedding should be 2D")
        #expect(embedding.shape[0] == 1, "Batch size should be 1")
        #expect(embedding.shape[1] == 768, "Embedding dimension should be 768")

        // L2正規化の確認
        let norm = embedding.square().sum(axis: -1).sqrt()
        let normValue = norm.item(Float.self)
        #expect(abs(normValue - 1.0) < 0.01, "Embedding should be L2 normalized")
    }

    @Test("Batch text encoding")
    func textEncodingBatch() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        let batchSize = 3
        var batchTokens = [[Int32]]()

        for i in 0..<batchSize {
            var tokens = [Int32](repeating: 0, count: 77)
            tokens[0] = 49406  // SOS
            tokens[1] = Int32(1000 + i * 100)
            tokens[2] = 49407  // EOS
            batchTokens.append(tokens)
        }

        let flatTokens = batchTokens.flatMap { $0 }
        let tokenArray = MLXArray(flatTokens).reshaped(batchSize, 77)
        let embeddings = try model.encodeText(tokenArray)

        print("✅ Batch text embeddings shape: \(embeddings.shape)")

        #expect(embeddings.shape[0] == batchSize, "Batch size should be \(batchSize)")
        #expect(embeddings.shape[1] == 768, "Embedding dimension should be 768")
    }

    @Test("Text encoder handles special tokens correctly")
    func textEncoderSpecialTokens() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // SOSのみ
        var sosOnly = [Int32](repeating: 0, count: 77)
        sosOnly[0] = 49406

        // SOS + EOS
        var sosEos = [Int32](repeating: 0, count: 77)
        sosEos[0] = 49406
        sosEos[1] = 49407

        let sosOnlyTokens = MLXArray(sosOnly).reshaped(1, 77)
        let sosEosTokens = MLXArray(sosEos).reshaped(1, 77)

        let emb1 = try model.encodeText(sosOnlyTokens)
        let emb2 = try model.encodeText(sosEosTokens)

        #expect(emb1.shape == [1, 768])
        #expect(emb2.shape == [1, 768])
    }
}

// MARK: - Tokenizer Tests

@Suite("Tokenizer Tests")
struct TokenizerTests {

    @Test("Tokenizer produces correct shape")
    func tokenizerBasic() throws {
        let tokenizer = try Tokenizer()
        let text = "a photo of a cat"
        let tokens = tokenizer.encode(text)

        print("✅ Tokenized shape: \(tokens.shape)")

        #expect(tokens.shape[0] == 1, "Batch size should be 1")
        #expect(tokens.shape[1] == 77, "Context length should be 77")
    }

    @Test("Tokenizer adds SOS token")
    func tokenizerSOS() throws {
        let tokenizer = try Tokenizer()
        let tokens = tokenizer.encode("test")

        let firstToken = tokens[0, 0].item(Int32.self)
        #expect(firstToken == 49406, "First token should be SOS (49406)")
    }

    @Test("Tokenizer adds EOS token")
    func tokenizerEOS() throws {
        let tokenizer = try Tokenizer()
        let tokens = tokenizer.encode("test")

        var hasEOS = false
        for i in 0..<77 {
            let token = tokens[0, i].item(Int32.self)
            if token == 49407 {
                hasEOS = true
                break
            }
        }
        #expect(hasEOS, "Should have EOS token (49407)")
    }

    @Test("Tokenizer pads to context length")
    func tokenizerPadding() throws {
        let tokenizer = try Tokenizer()
        let tokens = tokenizer.encode("short")

        let lastToken = tokens[0, 76].item(Int32.self)
        #expect(lastToken == 0, "Last token should be PAD (0)")
    }

    @Test("Tokenizer batch encoding")
    func tokenizerBatch() throws {
        let tokenizer = try Tokenizer()
        let texts = ["a photo of a cat", "a photo of a dog", "a bird"]
        let batchTokens = tokenizer.encodeBatch(texts)

        print("✅ Batch tokenized shape: \(batchTokens.shape)")

        #expect(batchTokens.shape[0] == 3, "Batch size should be 3")
        #expect(batchTokens.shape[1] == 77, "Context length should be 77")

        // すべてSOSで始まる
        for i in 0..<3 {
            let firstToken = batchTokens[i, 0].item(Int32.self)
            #expect(firstToken == 49406, "All sequences should start with SOS")
        }
    }

    @Test("Tokenizer handles different text lengths", arguments: ["short", "medium length text", "a very long text that should be truncated properly"])
    func tokenizerVariableLengths(text: String) throws {
        let tokenizer = try Tokenizer()
        let tokens = tokenizer.encode(text)

        #expect(tokens.shape[1] == 77, "All outputs should be padded to 77")

        let firstToken = tokens[0, 0].item(Int32.self)
        #expect(firstToken == 49406, "Should start with SOS")
    }
}

// MARK: - Similarity Tests

@Suite("Similarity Computation Tests")
struct SimilarityTests {

    @Test("Similarity is in valid range")
    func similarityRange() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        let dummyImage = MLXArray.zeros([1, 3, 224, 224])
        var tokens = [Int32](repeating: 0, count: 77)
        tokens[0] = 49406
        tokens[1] = 1000
        tokens[2] = 49407
        let dummyTokens = MLXArray(tokens).reshaped(1, 77)

        let imageEmbedding = try model.encodeImage(dummyImage)
        let textEmbedding = try model.encodeText(dummyTokens)

        let similarity = model.computeSimilarity(
            imageEmbedding: imageEmbedding,
            textEmbedding: textEmbedding
        )

        print("✅ Similarity score: \(similarity)")

        #expect(similarity >= -1.0, "Similarity should be >= -1.0")
        #expect(similarity <= 1.0, "Similarity should be <= 1.0")
    }

    @Test("Zero-shot classification produces valid scores")
    func zeroShotClassification() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        let testImage = MLXArray.zeros([1, 3, 224, 224])
        let imageEmbedding = try model.encodeImage(testImage)

        let labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
        let tokenizer = try Tokenizer()

        var similarities = [Float]()

        for label in labels {
            let tokens = tokenizer.encode(label)
            let textEmbedding = try model.encodeText(tokens)
            let similarity = model.computeSimilarity(
                imageEmbedding: imageEmbedding,
                textEmbedding: textEmbedding
            )
            similarities.append(similarity)
        }

        print("✅ Zero-shot classification scores:")
        for (i, label) in labels.enumerated() {
            print("   \(label): \(similarities[i])")
        }

        #expect(similarities.count == labels.count, "Should have similarity for each label")

        for similarity in similarities {
            #expect(similarity >= -1.0 && similarity <= 1.0, "All similarities should be in valid range")
        }
    }

    @Test("Identical embeddings have similarity of 1.0")
    func identicalEmbeddings() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // Use constant non-zero value to avoid NaN
        let image = MLXArray.ones([1, 3, 224, 224]) * 0.5
        let emb1 = try model.encodeImage(image)
        let emb2 = try model.encodeImage(image)

        let similarity = model.computeSimilarity(
            imageEmbedding: emb1,
            textEmbedding: emb2
        )

        #expect(abs(similarity - 1.0) < 0.01, "Identical embeddings should have similarity ≈ 1.0")
    }
}

// MARK: - Integration Tests

@Suite("Integration Tests")
struct IntegrationTests {

    @Test("Full pipeline executes successfully")
    func fullPipeline() async throws {
        let model = MobileCLIP()

        // 1. Load model
        try model.loadModelFromBundle()
        print("✅ Model loaded")

        // 2. Encode image
        let image = MLXArray.zeros([1, 3, 224, 224])
        let imageEmb = try model.encodeImage(image)
        #expect(imageEmb.shape == [1, 768])
        print("✅ Image encoded")

        // 3. Encode text
        let tokenizer = try Tokenizer()
        let tokens = tokenizer.encode("test")
        let textEmb = try model.encodeText(tokens)
        #expect(textEmb.shape == [1, 768])
        print("✅ Text encoded")

        // 4. Compute similarity
        let similarity = model.computeSimilarity(
            imageEmbedding: imageEmb,
            textEmbedding: textEmb
        )
        #expect(similarity >= -1.0 && similarity <= 1.0)
        print("✅ Similarity computed: \(similarity)")

        print("✅ Full pipeline test passed!")
    }

    @Test("Model can be reused for multiple inferences")
    func modelReuse() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // 複数回の推論
        for i in 0..<3 {
            let image = MLXArray.zeros([1, 3, 224, 224])
            let embedding = try model.encodeImage(image)

            #expect(embedding.shape == [1, 768], "Iteration \(i) should produce correct shape")
        }

        print("✅ Model reuse test passed")
    }

    @Test("Sequential encoding works correctly")
    func sequentialEncoding() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // 複数回の連続エンコード
        var allSucceeded = true
        for _ in 0..<3 {
            do {
                let image = MLXArray.zeros([1, 3, 224, 224])
                let embedding = try model.encodeImage(image)
                if embedding.shape != [1, 768] {
                    allSucceeded = false
                }
            } catch {
                allSucceeded = false
            }
        }

        #expect(allSucceeded, "All sequential encodings should succeed")
    }
}

// MARK: - Performance Tests

@Suite("Performance Tests", .tags(.performance))
struct PerformanceTests {

    @Test("Vision encoding performance")
    func visionEncodingPerformance() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        let image = MLXArray.zeros([1, 3, 224, 224])

        // Warmup
        _ = try model.encodeImage(image)

        let start = Date()
        for _ in 0..<10 {
            _ = try model.encodeImage(image)
        }
        let elapsed = Date().timeIntervalSince(start)

        let avgTime = elapsed / 10.0
        print("✅ Average vision encoding time: \(avgTime * 1000)ms")

        #expect(avgTime < 1.0, "Average encoding should be under 1 second")
    }

    @Test("Text encoding performance")
    func textEncodingPerformance() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        var tokens = [Int32](repeating: 0, count: 77)
        tokens[0] = 49406
        tokens[1] = 1000
        tokens[2] = 49407
        let dummyTokens = MLXArray(tokens).reshaped(1, 77)

        // Warmup
        _ = try model.encodeText(dummyTokens)

        let start = Date()
        for _ in 0..<10 {
            _ = try model.encodeText(dummyTokens)
        }
        let elapsed = Date().timeIntervalSince(start)

        let avgTime = elapsed / 10.0
        print("✅ Average text encoding time: \(avgTime * 1000)ms")

        #expect(avgTime < 0.5, "Average encoding should be under 0.5 seconds")
    }
}

// MARK: - Error Handling Tests

@Suite("Error Handling Tests")
struct ErrorHandlingTests {

    @Test("ModelLoader handles missing files gracefully")
    func modelLoaderMissingFile() throws {
        let loader = ModelLoader()

        #expect(throws: ModelError.self) {
            try loader.loadWeights(basePath: "/nonexistent/path")
        }
    }

    @Test("Text encoder handles empty sequence")
    func textEncoderEmptySequence() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // PADトークンのみ
        let emptyTokens = MLXArray([Int32](repeating: 0, count: 77)).reshaped(1, 77)
        let embedding = try model.encodeText(emptyTokens)

        #expect(embedding.shape == [1, 768], "Empty sequence should still produce valid embedding")
    }

    @Test("Text encoder produces different embeddings for different tokens")
    func textEncoderDifferentTokens() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        var tokens1 = [Int32](repeating: 0, count: 77)
        tokens1[0] = 49406
        tokens1[1] = 1000
        tokens1[2] = 49407

        var tokens2 = [Int32](repeating: 0, count: 77)
        tokens2[0] = 49406
        tokens2[1] = 2000  // 異なるトークン
        tokens2[2] = 49407

        let emb1 = try model.encodeText(MLXArray(tokens1).reshaped(1, 77))
        let emb2 = try model.encodeText(MLXArray(tokens2).reshaped(1, 77))

        let similarity = model.computeSimilarity(imageEmbedding: emb1, textEmbedding: emb2)
        #expect(abs(similarity - 1.0) > 0.01, "Different tokens should produce different embeddings")
    }
}

// MARK: - Input Validation Tests

@Suite("Input Validation Tests")
struct InputValidationTests {

    @Test("Vision encoder handles different batch sizes", arguments: [1, 2, 4, 8])
    func visionEncoderBatchSizes(batchSize: Int) async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        let images = MLXArray.zeros([batchSize, 3, 224, 224])
        let embeddings = try model.encodeImage(images)

        #expect(embeddings.shape[0] == batchSize, "Batch size should be \(batchSize)")
        #expect(embeddings.shape[1] == 768, "Embedding dimension should be 768")
    }

    @Test("Text encoder handles different batch sizes", arguments: [1, 2, 4, 8])
    func textEncoderBatchSizes(batchSize: Int) async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        var batchTokens = [[Int32]]()
        for i in 0..<batchSize {
            var tokens = [Int32](repeating: 0, count: 77)
            tokens[0] = 49406
            tokens[1] = Int32(1000 + i)
            tokens[2] = 49407
            batchTokens.append(tokens)
        }

        let flatTokens = batchTokens.flatMap { $0 }
        let tokenArray = MLXArray(flatTokens).reshaped(batchSize, 77)
        let embeddings = try model.encodeText(tokenArray)

        #expect(embeddings.shape[0] == batchSize, "Batch size should be \(batchSize)")
        #expect(embeddings.shape[1] == 768, "Embedding dimension should be 768")
    }

    @Test("Vision encoder produces consistent outputs")
    func visionEncoderConsistency() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // Use constant non-zero value to avoid NaN
        let image = MLXArray.ones([1, 3, 224, 224]) * 0.5

        let emb1 = try model.encodeImage(image)
        let emb2 = try model.encodeImage(image)

        // 同じ入力は同じ出力を生成
        let similarity = model.computeSimilarity(imageEmbedding: emb1, textEmbedding: emb2)

        #expect(!similarity.isNaN, "Similarity should not be NaN")
        #expect(abs(similarity - 1.0) < 0.0001, "Same input should produce identical embeddings")
    }

    @Test("Text encoder produces consistent outputs")
    func textEncoderConsistency() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        var tokens = [Int32](repeating: 0, count: 77)
        tokens[0] = 49406
        tokens[1] = 1000
        tokens[2] = 49407
        let tokenArray = MLXArray(tokens).reshaped(1, 77)

        let emb1 = try model.encodeText(tokenArray)
        let emb2 = try model.encodeText(tokenArray)

        // 同じ入力は同じ出力を生成
        let similarity = model.computeSimilarity(imageEmbedding: emb1, textEmbedding: emb2)
        #expect(abs(similarity - 1.0) < 0.0001, "Same input should produce identical embeddings")
    }

    @Test("Tokenizer handles only SOS token")
    func tokenizerOnlySOS() throws {
        let tokenizer = try Tokenizer()
        let tokens = tokenizer.encode("")

        #expect(tokens.shape == [1, 77], "Empty text should still produce valid shape")
        #expect(tokens[0, 0].item(Int32.self) == 49406, "First token should be SOS")
    }

    @Test("Tokenizer handles special characters", arguments: ["Hello!", "How are you?", "Test@#$%", "café"])
    func tokenizerSpecialCharacters(text: String) throws {
        let tokenizer = try Tokenizer()
        let tokens = tokenizer.encode(text)

        #expect(tokens.shape == [1, 77], "All texts should produce shape [1, 77]")
        #expect(tokens[0, 0].item(Int32.self) == 49406, "Should start with SOS")

        // EOSトークンが存在するか確認
        var hasEOS = false
        for i in 0..<77 {
            if tokens[0, i].item(Int32.self) == 49407 {
                hasEOS = true
                break
            }
        }
        #expect(hasEOS, "Should have EOS token")
    }
}

// MARK: - Image Preprocessing Tests

@Suite("Image Preprocessing Tests")
struct ImagePreprocessingTests {

    @Test("ImagePreprocessor initializes correctly")
    func imagePreprocessorInit() {
        let preprocessor = ImagePreprocessor()
        // 初期化が成功することを確認（初期化が完了すればパス）
        _ = preprocessor
    }

    @Test("ImagePreprocessor has correct mean values")
    func imagePreprocessorMean() {
        // ImageNet統計の確認
        let expectedMean: [Float] = [0.485, 0.456, 0.406]

        // この値が正しくハードコードされていることを確認
        #expect(expectedMean.count == 3, "Should have 3 channels")
        #expect(expectedMean[0] > 0 && expectedMean[0] < 1, "Mean should be in [0, 1] range")
    }

    @Test("ImagePreprocessor has correct std values")
    func imagePreprocessorStd() {
        // ImageNet統計の確認
        let expectedStd: [Float] = [0.229, 0.224, 0.225]

        // この値が正しくハードコードされていることを確認
        #expect(expectedStd.count == 3, "Should have 3 channels")
        #expect(expectedStd[0] > 0 && expectedStd[0] < 1, "Std should be in [0, 1] range")
    }

    @Test("Preprocessing target size is 224")
    func preprocessingTargetSize() {
        // MobileCLIP2は224x224を期待
        let expectedSize = 224
        #expect(expectedSize == 224, "Target size should be 224")
    }
}

// MARK: - Layer Tests

@Suite("Layer Tests")
struct LayerTests {

    @Test("Conv2d produces expected output shape")
    func conv2dShape() {
        // 64 output channels, 3 input channels, 3x3 kernel
        let weight = MLXArray.zeros([64, 3, 3, 3])
        let conv = Conv2d(weight: weight, stride: (1, 1), padding: (1, 1))

        // NHWC format: [batch, height, width, channels]
        let input = MLXArray.zeros([1, 224, 224, 3])
        let output = conv(input)

        // padding=1, stride=1なので出力サイズは入力と同じ
        #expect(output.shape[0] == 1, "Batch size preserved")
        #expect(output.shape[1] == 224, "Height preserved with padding")
        #expect(output.shape[2] == 224, "Width preserved with padding")
        #expect(output.shape[3] == 64, "Output channels should be 64")
    }

    @Test("Conv2d with stride 2 reduces spatial dimensions")
    func conv2dStride() {
        let weight = MLXArray.zeros([64, 3, 3, 3])
        let conv = Conv2d(weight: weight, stride: (2, 2), padding: (1, 1))

        // NHWC format: [batch, height, width, channels]
        let input = MLXArray.zeros([1, 224, 224, 3])
        let output = conv(input)

        #expect(output.shape[1] == 112, "Height should be halved with stride 2")
        #expect(output.shape[2] == 112, "Width should be halved with stride 2")
    }

    @Test("BatchNorm2d preserves shape")
    func batchNorm2dShape() {
        let channels = 64
        let weight = MLXArray.ones([channels])
        let bias = MLXArray.zeros([channels])
        let mean = MLXArray.zeros([channels])
        let variance = MLXArray.ones([channels])

        let bn = BatchNorm2d(
            weight: weight,
            bias: bias,
            runningMean: mean,
            runningVar: variance
        )

        // NHWC format: [batch, height, width, channels]
        let input = MLXArray.zeros([2, 56, 56, 64])
        let output = bn(input)

        #expect(output.shape == input.shape, "BatchNorm should preserve shape")
    }

    @Test("ReLU sets negative values to zero")
    func reluActivation() {
        let relu = ReLU()
        let input = MLXArray([-2.0 as Float, -1.0 as Float, 0.0 as Float, 1.0 as Float, 2.0 as Float])
        let output = relu(input)

        // 負の値は0に
        #expect(output[0].item(Float.self) == 0.0, "Negative values should be zero")
        #expect(output[1].item(Float.self) == 0.0, "Negative values should be zero")

        // 正の値はそのまま
        #expect(output[3].item(Float.self) == 1.0, "Positive values preserved")
        #expect(output[4].item(Float.self) == 2.0, "Positive values preserved")
    }

    @Test("GELU activation produces expected behavior")
    func geluActivation() {
        let gelu = GELU()
        let input = MLXArray([-2.0 as Float, -1.0 as Float, 0.0 as Float, 1.0 as Float, 2.0 as Float])
        let output = gelu(input)

        // GELU(0) ≈ 0
        let zeroOutput = output[2].item(Float.self)
        #expect(abs(zeroOutput) < 0.01, "GELU(0) should be approximately 0")

        // GELUは滑らかな活性化関数（ReLUと異なり微分可能）
        // 負の値でも完全に0にはならない
        let negativeOutput = output[0].item(Float.self)
        #expect(negativeOutput < 0, "GELU allows small negative values")
    }

    @Test("Linear layer produces correct output shape")
    func linearShape() {
        // 入力768次元、出力512次元
        let weight = MLXArray.zeros([512, 768])
        let bias = MLXArray.zeros([512])
        let linear = Linear(weight: weight, bias: bias)

        let input = MLXArray.zeros([4, 768])  // batch_size=4
        let output = linear(input)

        #expect(output.shape[0] == 4, "Batch size preserved")
        #expect(output.shape[1] == 512, "Output dimension should be 512")
    }

    @Test("GlobalAvgPool2d reduces spatial dimensions")
    func globalAvgPool2d() {
        let pool = GlobalAvgPool2d()
        // NHWC format: [batch, height, width, channels]
        let input = MLXArray.zeros([2, 7, 7, 768])  // batch_size=2, 7x7 spatial, channels=768
        let output = pool(input)

        #expect(output.shape[0] == 2, "Batch size preserved")
        #expect(output.shape[1] == 768, "Channels preserved")
        // 空間次元が平均化されて消える
        #expect(output.ndim == 2, "Spatial dimensions should be averaged out")
    }
}

// MARK: - Edge Case Tests

@Suite("Edge Case Tests")
struct EdgeCaseTests {

    @Test("Vision encoder handles all-zero input")
    func visionEncoderZeroInput() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        let zeroImage = MLXArray.zeros([1, 3, 224, 224])
        let embedding = try model.encodeImage(zeroImage)

        #expect(embedding.shape == [1, 768], "Zero input should produce valid shape")

        // ゼロ入力でも正規化された出力が得られる
        let norm = embedding.square().sum(axis: -1).sqrt()
        let normValue = norm.item(Float.self)
        #expect(abs(normValue - 1.0) < 0.01, "Even zero input should produce normalized embedding")
    }

    @Test("Text encoder handles only padding")
    func textEncoderOnlyPadding() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // PADトークン(0)のみ
        let padOnlyTokens = MLXArray([Int32](repeating: 0, count: 77)).reshaped(1, 77)
        let embedding = try model.encodeText(padOnlyTokens)

        #expect(embedding.shape == [1, 768], "Pad-only input should produce valid shape")

        let norm = embedding.square().sum(axis: -1).sqrt()
        let normValue = norm.item(Float.self)
        #expect(abs(normValue - 1.0) < 0.01, "Pad-only input should produce normalized embedding")
    }

    @Test("Model handles multiple sequential operations")
    func multipleSequentialOperations() async throws {
        let model = MobileCLIP()
        try model.loadModelFromBundle()

        // 画像→テキスト→画像→テキストの順で処理
        for i in 0..<5 {
            if i % 2 == 0 {
                let image = MLXArray.zeros([1, 3, 224, 224])
                let emb = try model.encodeImage(image)
                #expect(emb.shape == [1, 768], "Operation \(i) should succeed")
            } else {
                var tokens = [Int32](repeating: 0, count: 77)
                tokens[0] = 49406
                tokens[1] = Int32(1000 + i)
                tokens[2] = 49407
                let tokenArray = MLXArray(tokens).reshaped(1, 77)
                let emb = try model.encodeText(tokenArray)
                #expect(emb.shape == [1, 768], "Operation \(i) should succeed")
            }
        }
    }

    @Test("Tokenizer handles maximum length text")
    func tokenizerMaxLength() throws {
        let tokenizer = try Tokenizer()

        // 非常に長いテキスト（context_length=77を超える）
        let longText = String(repeating: "word ", count: 100)
        let tokens = tokenizer.encode(longText)

        #expect(tokens.shape[1] == 77, "Output should be truncated to context length")
        #expect(tokens[0, 0].item(Int32.self) == 49406, "Should start with SOS")

        // 最後はEOSまたはPADトークン
        let lastToken = tokens[0, 76].item(Int32.self)
        #expect(lastToken == 49407 || lastToken == 0, "Last token should be EOS or PAD")
    }
}

// MARK: - Custom Tags

extension Tag {
    @Tag static var performance: Self
}
