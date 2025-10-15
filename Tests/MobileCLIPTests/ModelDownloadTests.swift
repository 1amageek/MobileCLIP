import Testing
import Foundation
import MLX
import MLXLMCommon
import Hub
@testable import MobileCLIP

@Suite("Model Download Tests")
struct ModelDownloadTests {

    @Test("MLXLMCommon.downloadModel() downloads from Hugging Face Hub", .timeLimit(.minutes(5)))
    func downloadModelFromHub() async throws {
        print("🔍 Testing MLXLMCommon.downloadModel() functionality")
        print("   Repository: 1amageek/MobileCLIP2-S4")

        let configuration = ModelConfiguration(id: "1amageek/MobileCLIP2-S4")

        let modelDirectory = try await MLXLMCommon.downloadModel(
            hub: .shared,
            configuration: configuration,
            progressHandler: { progress in
                let percentage = Int(progress.fractionCompleted * 100)
                if percentage % 10 == 0 {
                    print("   Download progress: \(percentage)%")
                }
            }
        )

        print("✅ Download completed")
        print("   Model directory: \(modelDirectory.path)")

        // Verify directory exists
        #expect(FileManager.default.fileExists(atPath: modelDirectory.path),
                "Model directory should exist at \(modelDirectory.path)")

        // List downloaded files
        let contents = try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: []
        )

        print("📁 Downloaded files:")
        for file in contents {
            let attributes = try file.resourceValues(forKeys: [.fileSizeKey])
            let size = attributes.fileSize ?? 0
            let sizeInMB = Double(size) / (1024 * 1024)
            print("   - \(file.lastPathComponent) (\(String(format: "%.2f", sizeInMB)) MB)")
        }

        // Verify safetensors file exists
        let safetensorsFiles = contents.filter { $0.pathExtension == "safetensors" }
        #expect(!safetensorsFiles.isEmpty,
                "At least one .safetensors file should be present")

        if let safetensors = safetensorsFiles.first {
            print("✅ Found safetensors file: \(safetensors.lastPathComponent)")

            // Verify it's the correct model
            #expect(safetensors.lastPathComponent.contains("MobileCLIP2-S4"),
                    "Safetensors file should be MobileCLIP2-S4 model")
        }

        print("✅ Test completed successfully")
    }

    @Test("MobileCLIP.load() successfully loads model from Hub")
    func loadModelFromHub() async throws {
        print("🔍 Testing MobileCLIP.load() with Hugging Face Hub")

        let model = MobileCLIP()

        try await model.load(
            configuration: ModelConfiguration(id: "1amageek/MobileCLIP2-S4"),
            progressHandler: { progress in
                let percentage = Int(progress.fractionCompleted * 100)
                if percentage % 10 == 0 {
                    print("   Loading progress: \(percentage)%")
                }
            }
        )

        print("✅ Model loaded successfully")

        // Verify model is functional
        let dummyImage = MLXArray.ones([1, 3, 224, 224]) * 0.5
        let output = try model.encodeImage(dummyImage)

        print("   Output shape: \(output.shape)")

        #expect(output.shape[0] == 1, "Batch size should be 1")
        #expect(output.shape[1] == 768, "Embedding dimension should be 768")

        print("✅ Model is functional")
    }

    @Test("Downloaded model is cached and reusable")
    func modelCaching() async throws {
        print("🔍 Testing model caching functionality")

        // First load - may download
        print("   First load...")
        let model1 = MobileCLIP()
        let start1 = Date()
        try await model1.load(
            configuration: ModelConfiguration(id: "1amageek/MobileCLIP2-S4")
        )
        let duration1 = Date().timeIntervalSince(start1)
        print("   First load duration: \(String(format: "%.2f", duration1))s")

        // Second load - should use cache
        print("   Second load (from cache)...")
        let model2 = MobileCLIP()
        let start2 = Date()
        try await model2.load(
            configuration: ModelConfiguration(id: "1amageek/MobileCLIP2-S4")
        )
        let duration2 = Date().timeIntervalSince(start2)
        print("   Second load duration: \(String(format: "%.2f", duration2))s")

        // Second load should be faster (using cache)
        print("   Speedup: \(String(format: "%.2f", duration1 / duration2))x")

        // Both models should produce same output
        let input = MLXArray.ones([1, 3, 224, 224]) * 0.5
        let output1 = try model1.encodeImage(input)
        let output2 = try model2.encodeImage(input)

        let diff = (output1 - output2).abs().sum()
        let diffValue = diff.item(Float.self)

        print("   Output difference: \(diffValue)")

        #expect(diffValue < 1e-6, "Both models should produce identical outputs")
        print("✅ Caching works correctly")
    }

    @Test("Check Hugging Face cache location")
    func checkCacheLocation() async throws {
        print("🔍 Checking Hugging Face cache")

        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let cacheDir = homeDir.appendingPathComponent(".cache/huggingface/hub")

        print("   Cache directory: \(cacheDir.path)")

        if FileManager.default.fileExists(atPath: cacheDir.path) {
            print("✅ Cache directory exists")

            let contents = try FileManager.default.contentsOfDirectory(
                at: cacheDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: []
            )

            print("   Cached repositories:")
            for item in contents where item.lastPathComponent.hasPrefix("models--") {
                print("   - \(item.lastPathComponent)")
            }

            // Check for our model
            let ourModel = contents.first {
                $0.lastPathComponent.contains("1amageek") &&
                $0.lastPathComponent.contains("MobileCLIP2-S4")
            }

            if let modelDir = ourModel {
                print("   ✅ Found our model cache: \(modelDir.lastPathComponent)")
            }
        } else {
            print("   ℹ️ Cache directory doesn't exist yet")
        }
    }
}
