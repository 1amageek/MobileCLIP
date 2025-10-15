import Foundation
import MLX

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Image preprocessing for MobileCLIP2
///
/// Preprocessing steps:
/// 1. Center crop to square (if needed)
/// 2. Resize to target size (default: 224x224)
/// 3. Convert to [0, 1] range
/// 4. Normalize with ImageNet statistics
///    - mean: [0.485, 0.456, 0.406]
///    - std: [0.229, 0.224, 0.225]
/// 5. Convert to [C, H, W] format
public class ImagePreprocessor {

    private let targetSize: Int
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std: [Float] = [0.229, 0.224, 0.225]
    private let applyCenterCrop: Bool

    /// Initialize preprocessor
    /// - Parameters:
    ///   - targetSize: Target image size (default: 224 for MobileCLIP-S models, 256 for larger models)
    ///   - centerCrop: Whether to apply center crop (default: false)
    public init(targetSize: Int = 224, centerCrop: Bool = false) {
        self.targetSize = targetSize
        self.applyCenterCrop = centerCrop
    }

    #if canImport(UIKit)
    /// UIImageをMLXArrayに変換
    /// - Parameter image: 入力画像
    /// - Returns: [1, 3, targetSize, targetSize] のテンソル
    public func preprocess(_ image: UIImage) -> MLXArray {
        var processedImage = image

        // 1. Center crop to square (if enabled)
        if applyCenterCrop {
            processedImage = centerCrop(image)
        }

        // 2. Resize to target size
        guard let resized = resizeImage(processedImage, to: CGSize(width: targetSize, height: targetSize)) else {
            print("⚠️ Failed to resize image, using zeros")
            return MLXArray.zeros([1, 3, targetSize, targetSize])
        }

        // 3. Extract pixel data
        guard let pixelData = extractPixelData(from: resized) else {
            print("⚠️ Failed to extract pixels, using zeros")
            return MLXArray.zeros([1, 3, targetSize, targetSize])
        }

        // 4. Convert to MLXArray and normalize
        return normalizePixels(pixelData)
    }

    /// Center crop image to square
    private func centerCrop(_ image: UIImage) -> UIImage {
        guard let cgImage = image.cgImage else { return image }

        let width = cgImage.width
        let height = cgImage.height

        // Already square
        if width == height {
            return image
        }

        // Calculate crop rect
        let cropSize = min(width, height)
        let x = (width - cropSize) / 2
        let y = (height - cropSize) / 2

        guard let cropped = cgImage.cropping(to: CGRect(x: x, y: y, width: cropSize, height: cropSize)) else {
            return image
        }

        return UIImage(cgImage: cropped, scale: image.scale, orientation: image.imageOrientation)
    }

    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }

        image.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    private func extractPixelData(from image: UIImage) -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }

        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8

        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBA to RGB float values [0, 1]
        var floatPixels = [Float]()
        floatPixels.reserveCapacity(width * height * 3)

        for i in stride(from: 0, to: pixelData.count, by: bytesPerPixel) {
            let r = Float(pixelData[i]) / 255.0
            let g = Float(pixelData[i + 1]) / 255.0
            let b = Float(pixelData[i + 2]) / 255.0

            floatPixels.append(r)
            floatPixels.append(g)
            floatPixels.append(b)
        }

        return floatPixels
    }
    #endif

    #if canImport(AppKit)
    /// NSImageをMLXArrayに変換
    /// - Parameter image: 入力画像
    /// - Returns: [1, 3, targetSize, targetSize] のテンソル
    public func preprocess(_ image: NSImage) -> MLXArray {
        // Convert NSImage to CGImage
        guard var cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            print("⚠️ Failed to convert NSImage to CGImage, using zeros")
            return MLXArray.zeros([1, 3, targetSize, targetSize])
        }

        // Center crop (if enabled)
        if applyCenterCrop {
            cgImage = centerCrop(cgImage)
        }

        // Resize
        guard let resized = resizeImage(cgImage, to: CGSize(width: targetSize, height: targetSize)) else {
            print("⚠️ Failed to resize image, using zeros")
            return MLXArray.zeros([1, 3, targetSize, targetSize])
        }

        // Extract pixels
        guard let pixelData = extractPixelData(from: resized) else {
            print("⚠️ Failed to extract pixels, using zeros")
            return MLXArray.zeros([1, 3, targetSize, targetSize])
        }

        // Normalize
        return normalizePixels(pixelData)
    }

    /// Center crop CGImage to square
    private func centerCrop(_ cgImage: CGImage) -> CGImage {
        let width = cgImage.width
        let height = cgImage.height

        // Already square
        if width == height {
            return cgImage
        }

        // Calculate crop rect
        let cropSize = min(width, height)
        let x = (width - cropSize) / 2
        let y = (height - cropSize) / 2

        guard let cropped = cgImage.cropping(to: CGRect(x: x, y: y, width: cropSize, height: cropSize)) else {
            return cgImage
        }

        return cropped
    }

    private func resizeImage(_ cgImage: CGImage, to size: CGSize) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()
    }

    private func extractPixelData(from cgImage: CGImage) -> [Float]? {
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8

        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBA to RGB float values [0, 1]
        var floatPixels = [Float]()
        floatPixels.reserveCapacity(width * height * 3)

        for i in stride(from: 0, to: pixelData.count, by: bytesPerPixel) {
            let r = Float(pixelData[i]) / 255.0
            let g = Float(pixelData[i + 1]) / 255.0
            let b = Float(pixelData[i + 2]) / 255.0

            floatPixels.append(r)
            floatPixels.append(g)
            floatPixels.append(b)
        }

        return floatPixels
    }
    #endif

    /// ピクセルデータを正規化してMLXArrayに変換
    /// - Parameter pixels: [H*W*3] のピクセル配列（RGB順）
    /// - Returns: [1, 3, H, W] の正規化されたテンソル
    private func normalizePixels(_ pixels: [Float]) -> MLXArray {
        let height = targetSize
        let width = targetSize

        // Separate into R, G, B channels
        var rChannel = [Float]()
        var gChannel = [Float]()
        var bChannel = [Float]()

        for i in stride(from: 0, to: pixels.count, by: 3) {
            rChannel.append(pixels[i])
            gChannel.append(pixels[i + 1])
            bChannel.append(pixels[i + 2])
        }

        // Normalize each channel with ImageNet statistics
        let normalizedR = rChannel.map { ($0 - mean[0]) / std[0] }
        let normalizedG = gChannel.map { ($0 - mean[1]) / std[1] }
        let normalizedB = bChannel.map { ($0 - mean[2]) / std[2] }

        // Create MLXArray [3, H, W]
        let rArray = MLXArray(normalizedR).reshaped(height, width)
        let gArray = MLXArray(normalizedG).reshaped(height, width)
        let bArray = MLXArray(normalizedB).reshaped(height, width)

        // Stack channels
        let imageArray = MLX.stacked([rArray, gArray, bArray], axis: 0)

        // Add batch dimension [1, 3, H, W]
        return imageArray.expandedDimensions(axis: 0)
    }

    /// 複数の画像をバッチで前処理
    /// - Parameter images: 入力画像の配列
    /// - Returns: [batch_size, 3, 224, 224] のテンソル
    #if canImport(UIKit)
    public func preprocessBatch(_ images: [UIImage]) -> MLXArray {
        let preprocessed = images.map { preprocess($0) }

        // Stack into batch
        return MLX.stacked(preprocessed, axis: 0).squeezed(axis: 1)
    }
    #endif

    #if canImport(AppKit)
    public func preprocessBatch(_ images: [NSImage]) -> MLXArray {
        let preprocessed = images.map { preprocess($0) }

        // Stack into batch
        return MLX.stacked(preprocessed, axis: 0).squeezed(axis: 1)
    }
    #endif
}

// MARK: - Convenience Extensions

#if canImport(UIKit)
extension UIImage {
    /// UIImageをMobileCLIP2用に前処理
    public func preprocessForCLIP() -> MLXArray {
        let preprocessor = ImagePreprocessor()
        return preprocessor.preprocess(self)
    }
}
#endif

#if canImport(AppKit)
extension NSImage {
    /// NSImageをMobileCLIP2用に前処理
    public func preprocessForCLIP() -> MLXArray {
        let preprocessor = ImagePreprocessor()
        return preprocessor.preprocess(self)
    }
}
#endif
