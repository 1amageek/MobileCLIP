import Foundation
import MLX
import MLXNN

/// 畳み込み層
public struct Conv2d {
    let weight: MLXArray
    let bias: MLXArray?
    let stride: (Int, Int)
    let padding: (Int, Int)
    let groups: Int

    public init(weight: MLXArray, bias: MLXArray? = nil, stride: (Int, Int) = (1, 1), padding: (Int, Int) = (0, 0), groups: Int? = nil) {
        // PyTorch weight format: (out_channels, in_channels_per_group, kernel_h, kernel_w)
        let inChannelsPerGroup = weight.shape[1]

        self.bias = bias
        self.stride = stride
        self.padding = padding

        // Determine groups
        // For depthwise convolution: inChannelsPerGroup == 1
        // We need to defer group calculation to runtime when we know input channels
        if let groups = groups {
            self.groups = groups
        } else if inChannelsPerGroup == 1 {
            // Depthwise: groups will be determined at runtime based on input channels
            self.groups = -1  // Marker for depthwise
        } else {
            self.groups = 1
        }

        // Transform weight to MLX format: (out, h, w, in_per_group)
        self.weight = weight.transposed(0, 2, 3, 1)
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Use MLX's native grouped/depthwise convolution
        // Input shape: [batch, height, width, channels]
        // Weight shape: [out_channels, kernel_h, kernel_w, in_channels_per_group]

        // Determine actual groups at runtime
        let actualGroups: Int
        if groups == -1 {
            // Depthwise: groups = input_channels
            actualGroups = input.shape[3]
        } else {
            actualGroups = groups
        }

        var output = MLX.conv2d(
            input,
            weight,
            stride: IntOrPair(stride),
            padding: IntOrPair(padding),
            groups: actualGroups
        )

        if let bias = bias {
            let biasReshaped = bias.reshaped(1, 1, 1, -1)
            output = output + biasReshaped
        }

        // Do NOT call eval() here - let MLX optimize the computation graph
        return output
    }
}

/// Batch Normalization層
public struct BatchNorm2d {
    let weight: MLXArray
    let bias: MLXArray
    let runningMean: MLXArray
    let runningVar: MLXArray
    let eps: Float

    // Cache reshaped parameters for better performance
    private let meanReshaped: MLXArray
    private let varReshaped: MLXArray
    private let weightReshaped: MLXArray
    private let biasReshaped: MLXArray

    public init(
        weight: MLXArray,
        bias: MLXArray,
        runningMean: MLXArray,
        runningVar: MLXArray,
        eps: Float = 1e-2
    ) {
        self.weight = weight
        self.bias = bias
        self.runningMean = runningMean
        self.runningVar = runningVar
        self.eps = eps

        // Pre-compute reshaped parameters for NHWC format [1, 1, 1, channels]
        self.meanReshaped = runningMean.reshaped(1, 1, 1, -1)
        // Clamp variance to minimum 1e-3 to prevent value explosion
        // This is necessary because some BatchNorms have running_var as low as 6e-45
        self.varReshaped = maximum(runningVar, MLXArray(1e-3)).reshaped(1, 1, 1, -1)
        self.weightReshaped = weight.reshaped(1, 1, 1, -1)
        self.biasReshaped = bias.reshaped(1, 1, 1, -1)
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // BatchNorm in inference mode for NHWC format
        // Input shape: [batch, height, width, channels]
        // Use pre-computed reshaped parameters

        // Standard BatchNorm formula: (x - mean) / sqrt(var + eps)
        // Add eps for numerical stability
        let normalized = (input - meanReshaped) / sqrt(varReshaped + eps)

        // Apply affine transformation
        let output = normalized * weightReshaped + biasReshaped

        // Do NOT call eval() here - let MLX optimize the computation graph
        return output
    }
}

/// ReLU活性化関数
public struct ReLU {
    public init() {}

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Use MLX's optimized ReLU (free function)
        return relu(input)
    }
}

/// GELU活性化関数
public struct GELU {
    public init() {}

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Use MLX's optimized GELU (free function)
        // geluFastApproximate uses sigmoid approximation for best performance
        return geluFastApproximate(input)
    }
}

/// LayerNorm正規化
public struct LayerNorm {
    let weight: MLXArray
    let bias: MLXArray
    let eps: Float

    public init(weight: MLXArray, bias: MLXArray, eps: Float = 1e-5) {
        self.weight = weight
        self.bias = bias
        self.eps = eps
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let mean = input.mean(axis: -1, keepDims: true)
        let variance = ((input - mean) * (input - mean)).mean(axis: -1, keepDims: true)

        // Use scalar eps directly - MLX will handle Float32 conversion
        let normalized = (input - mean) / sqrt(variance + eps)

        return normalized * weight + bias
    }
}

/// 線形層（全結合層）
public struct Linear {
    let weight: MLXArray
    let bias: MLXArray?

    public init(weight: MLXArray, bias: MLXArray? = nil) {
        self.weight = weight
        self.bias = bias
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var output = matmul(input, weight.T)

        if let bias = bias {
            output = output + bias
        }

        // Do NOT call eval() here - let MLX optimize the computation graph
        return output
    }
}

/// Global Average Pooling
public struct GlobalAvgPool2d {
    public init() {}

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // input shape: [batch, height, width, channels]
        // output shape: [batch, channels]
        return input.mean(axes: [1, 2])
    }
}

/// Adaptive Average Pooling
public struct AdaptiveAvgPool2d {
    let outputSize: (Int, Int)

    public init(outputSize: (Int, Int)) {
        self.outputSize = outputSize
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // 簡易実装: outputSize = (1, 1) の場合は GlobalAvgPool2d と同じ
        if outputSize == (1, 1) {
            return GlobalAvgPool2d()(input)
        }

        // TODO: 一般的なサイズの実装
        fatalError("AdaptiveAvgPool2d: Only (1,1) is currently supported")
    }
}
