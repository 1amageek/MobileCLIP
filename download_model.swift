#!/usr/bin/env swift
import Foundation

// Swift script to download the model before running tests
// This ensures the model is cached locally

@main
struct DownloadModel {
    static func main() async throws {
        print("🔄 Downloading MobileCLIP2-S4 model from Hugging Face...")
        print("   Repository: 1amageek/MobileCLIP2-S4")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "swift", "run",
            "-c", "release",
            "--package-path", "/Users/1amageek/Desktop/MobileCLIP",
            "download-model"
        ]

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus == 0 {
            print("✅ Model downloaded successfully!")
        } else {
            print("❌ Failed to download model")
            exit(1)
        }
    }
}
