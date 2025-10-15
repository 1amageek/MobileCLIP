#!/usr/bin/env swift

import Foundation
import MLXLMCommon
import Hub

@main
struct TestDownload {
    static func main() async throws {
        print("🔍 Testing MLXLMCommon.downloadModel() functionality")
        print("   Repository: 1amageek/MobileCLIP2-S4\n")

        let configuration = ModelConfiguration(id: "1amageek/MobileCLIP2-S4")

        do {
            print("📥 Starting download...")
            let modelDirectory = try await MLXLMCommon.downloadModel(
                hub: .shared,
                configuration: configuration,
                progressHandler: { progress in
                    let percentage = Int(progress.fractionCompleted * 100)
                    print("   Progress: \(percentage)%")
                }
            )

            print("\n✅ Download successful!")
            print("   Model directory: \(modelDirectory.path)")

            // List downloaded files
            let contents = try FileManager.default.contentsOfDirectory(
                at: modelDirectory,
                includingPropertiesForKeys: [.fileSizeKey],
                options: []
            )

            print("\n📁 Downloaded files:")
            for file in contents {
                let attributes = try file.resourceValues(forKeys: [.fileSizeKey])
                let size = attributes.fileSize ?? 0
                let sizeInMB = Double(size) / (1024 * 1024)
                print("   - \(file.lastPathComponent) (\(String(format: "%.2f", sizeInMB)) MB)")
            }

            // Check for safetensors file
            if let safetensors = contents.first(where: { $0.pathExtension == "safetensors" }) {
                print("\n✅ Found safetensors file: \(safetensors.lastPathComponent)")
            } else {
                print("\n❌ No safetensors file found!")
            }

        } catch {
            print("\n❌ Download failed: \(error)")

            // Check if it's an offline mode error
            if let urlError = error as? URLError {
                print("   URL Error: \(urlError.localizedDescription)")
                print("   Code: \(urlError.errorCode)")
            }

            throw error
        }
    }
}
