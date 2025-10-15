import Foundation
import Hub

/// Downloads MobileCLIP models from Hugging Face
public actor ModelDownloader {

    /// Configuration for a MobileCLIP model on Hugging Face
    public struct ModelConfig {
        public let repoID: String
        public let filename: String

        public init(repoID: String, filename: String) {
            self.repoID = repoID
            self.filename = filename
        }

        /// Default MobileCLIP2-S4 configuration
        public static let mobileCLIP2S4 = ModelConfig(
            repoID: "1amageek/MobileCLIP2-S4",
            filename: "MobileCLIP2-S4.safetensors"
        )
    }

    private let hubApi: HubApi

    public init(hubApi: HubApi = .shared) {
        self.hubApi = hubApi
    }

    /// Download model from Hugging Face
    /// - Parameter config: Model configuration
    /// - Returns: URL to the downloaded model file
    public func download(
        config: ModelConfig,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> URL {
        let repo = Hub.Repo(id: config.repoID)

        // Download using HubApi.snapshot
        let modelDirectory = try await hubApi.snapshot(
            from: repo,
            matching: [config.filename]
        ) { progress in
            progressHandler(progress)
        }

        // Return path to the specific file
        let modelFile = modelDirectory.appending(path: config.filename)

        // Verify file exists
        guard FileManager.default.fileExists(atPath: modelFile.path) else {
            throw ModelDownloadError.fileNotFound(modelFile.path)
        }

        return modelFile
    }
}

/// Errors that can occur during model downloading
public enum ModelDownloadError: Error, LocalizedError {
    case fileNotFound(String)
    case downloadFailed(Error)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Model file not found at: \(path)"
        case .downloadFailed(let error):
            return "Failed to download model: \(error.localizedDescription)"
        }
    }
}
