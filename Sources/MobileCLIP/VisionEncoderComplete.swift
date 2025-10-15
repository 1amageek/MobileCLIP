import Foundation
import MLX
import MLXNN

/// MobileCLIP2-S4 Complete Vision Encoder
///
/// 正確なアーキテクチャ:
/// - 5 stages with [2, 12, 24, 4, 4] blocks
/// - Each block contains Token Mixer + MLP
/// - Stem with 3 layers
/// - Head with projection
public class VisionEncoderComplete {

    private let weights: [String: MLXArray]
    private let stageBlocks: [Int] = [2, 12, 24, 4, 4] // Stage 0-4

    public init(weights: [String: MLXArray]) {
        self.weights = weights
    }

    // MARK: - Main Encode Function

    /// 画像を埋め込みベクトルにエンコード
    /// - Parameter image: [batch, channels, height, width] の画像テンソル
    /// - Returns: [batch, embed_dim] の埋め込みベクトル
    public func encode(_ image: MLXArray) throws -> MLXArray {
        // MLXはNHWC形式を期待するため、NCHWからNHWCに変換
        // [batch, channels, height, width] → [batch, height, width, channels]
        var x = image.transposed(0, 2, 3, 1)

        // 1. Stem layers (3 blocks)
        x = try stem(x)

        // 2. Stage 0 (2 blocks)
        x = try processStage(x, stage: 0)

        // 3. Stage 1 (12 blocks with downsample)
        x = try processStage(x, stage: 1)

        // 4. Stage 2 (24 blocks with downsample)
        x = try processStage(x, stage: 2)

        // 5. Stage 3 (4 blocks with downsample)
        x = try processStage(x, stage: 3)

        // 6. Stage 4 (4 blocks with downsample)
        x = try processStage(x, stage: 4)

        // 7. Final Conv (2048 -> 4096 with SE)
        x = try finalConv(x)

        // 8. Head (projection)
        x = try head(x)

        // Evaluate the entire computation graph once at the end
        eval(x)

        return x
    }

    // MARK: - Stem Layers

    private func stem(_ x: MLXArray) throws -> MLXArray {
        var output = x

        // Stem consists of 3 conv blocks
        for i in 0..<3 {
            output = try stemBlock(output, index: i)
        }

        return output
    }

    private func stemBlock(_ x: MLXArray, index: Int) throws -> MLXArray {
        let prefix = "module.visual.trunk.stem.\(index)"

        // Identity branch (if exists) - for residual
        var identity: MLXArray? = nil
        if weights["\(prefix).identity.weight"] != nil {
            let identityBN = try loadBatchNorm(prefix: "\(prefix).identity")
            identity = identityBN(x)
        }

        // Conv branch
        let convPrefix = "\(prefix).conv_kxk.0"

        guard let convWeight = weights["\(convPrefix).conv.weight"] else {
            throw ModelError.weightNotFound("\(convPrefix).conv.weight")
        }

        let bn = try loadBatchNorm(prefix: "\(convPrefix).bn")

        // Determine stride and padding based on kernel size
        // stem.0 and stem.1: 3x3 conv with stride=2
        // stem.2: 1x1 conv with stride=1 (has residual connection)
        let kernelH = convWeight.shape[2]
        let kernelW = convWeight.shape[3]
        let stride: (Int, Int)
        let padding: (Int, Int)

        if kernelH == 1 && kernelW == 1 {
            // 1x1 pointwise conv
            stride = (1, 1)
            padding = (0, 0)
        } else {
            // 3x3 conv
            stride = (2, 2)
            padding = (1, 1)
        }

        // Conv -> BN -> ReLU
        let conv = Conv2d(weight: convWeight, stride: stride, padding: padding)
        var output = conv(x)
        output = bn(output)
        output = ReLU()(output)

        // Add identity if exists
        if let identity = identity {
            output = output + identity
        }

        // Do NOT call eval() here - let MLX optimize the computation graph
        return output
    }

    // MARK: - Stage Processing

    private func processStage(_ x: MLXArray, stage: Int) throws -> MLXArray {
        var output = x

        // Downsample (if exists for stage > 0)
        if stage > 0, weights["\(stagePrefix(stage)).downsample.proj.0.large_conv.conv.weight"] != nil {
            output = try downsample(output, stage: stage)
        }

        // Process all blocks in this stage
        let numBlocks = stageBlocks[stage]
        for blockIdx in 0..<numBlocks {
            output = try stageBlock(output, stage: stage, block: blockIdx)
        }

        return output
    }

    private func downsample(_ x: MLXArray, stage: Int) throws -> MLXArray {
        let prefix = "\(stagePrefix(stage)).downsample.proj.0"

        // Downsample proj.0: large_conv + small_conv (parallel paths)
        var output: MLXArray?

        // Large conv path (7x7 depthwise, stride=2)
        if let largeWeight = weights["\(prefix).large_conv.conv.weight"] {
            let largeBN = try loadBatchNorm(prefix: "\(prefix).large_conv.bn")
            let kernelH = largeWeight.shape[2]
            let padding = (kernelH / 2, kernelH / 2)
            let largeConv = Conv2d(weight: largeWeight, stride: (2, 2), padding: padding)
            var largeOut = largeConv(x)
            largeOut = largeBN(largeOut)

            output = largeOut
        }

        // Small conv path (3x3 depthwise, stride=2)
        if let smallWeight = weights["\(prefix).small_conv.conv.weight"] {
            let smallBN = try loadBatchNorm(prefix: "\(prefix).small_conv.bn")
            let smallConv = Conv2d(weight: smallWeight, stride: (2, 2), padding: (1, 1))
            var smallOut = smallConv(x)
            smallOut = smallBN(smallOut)

            if let existing = output {
                output = existing + smallOut
            } else {
                output = smallOut
            }
        }

        guard var result = output else {
            return x // No downsample
        }

        // Downsample proj.1: 1x1 conv + identity (similar to stem block)
        let proj1Prefix = "\(stagePrefix(stage)).downsample.proj.1"

        // Identity branch
        var identity: MLXArray? = nil
        if weights["\(proj1Prefix).identity.weight"] != nil {
            let identityBN = try loadBatchNorm(prefix: "\(proj1Prefix).identity")
            identity = identityBN(result)
        }

        // Conv branch
        if let convWeight = weights["\(proj1Prefix).conv_kxk.0.conv.weight"] {
            let convBN = try loadBatchNorm(prefix: "\(proj1Prefix).conv_kxk.0.bn")
            let conv = Conv2d(weight: convWeight, padding: (0, 0)) // 1x1 conv
            result = conv(result)
            result = convBN(result)
            result = ReLU()(result)

            // Add identity if exists
            if let identity = identity {
                result = result + identity
            }
        }

        return result
    }

    private func stageBlock(_ x: MLXArray, stage: Int, block: Int) throws -> MLXArray {
        let prefix = "\(stagePrefix(stage)).blocks.\(block)"

        // Check if this is an attention-based block (stages 3-4) or conv-based (stages 0-2)
        let isAttentionBlock = weights["\(prefix).token_mixer.qkv.weight"] != nil

        var output: MLXArray
        if isAttentionBlock {
            // Attention-based block (Transformer)
            output = try transformerBlock(x, prefix: prefix)
        } else {
            // Conv-based block
            output = x

            // 1. Token Mixer
            let tokenMixerOutput = try tokenMixer(x, prefix: "\(prefix).token_mixer")

            // Layer scale for token mixer
            if let gamma = weights["\(prefix).token_mixer.layer_scale.gamma"] {
                // Reshape gamma from [C, 1, 1] to [1, 1, 1, C] for NHWC format
                let gammaReshaped = gamma.reshaped(1, 1, 1, -1)
                output = output + gammaReshaped * tokenMixerOutput
            } else {
                output = output + tokenMixerOutput
            }

            // 2. MLP (Channel Mixer)
            let residual = output
            let mlpOutput = try mlp(output, prefix: "\(prefix).mlp")

            // Layer scale for MLP
            if let gamma = weights["\(prefix).layer_scale.gamma"] {
                // Reshape gamma from [C, 1, 1] to [1, 1, 1, C] for NHWC format
                let gammaReshaped = gamma.reshaped(1, 1, 1, -1)
                output = residual + gammaReshaped * mlpOutput
            } else {
                output = residual + mlpOutput
            }
        }

        // Do NOT call eval() here - let MLX optimize the computation graph
        return output
    }

    // MARK: - Token Mixer

    private func tokenMixer(_ x: MLXArray, prefix: String) throws -> MLXArray {
        // Norm
        let norm = try loadBatchNorm(prefix: "\(prefix).norm.identity")
        let output = norm(x)

        // Mixer - Identity branch
        let mixerIdentity = try loadBatchNorm(prefix: "\(prefix).mixer.identity")
        var mixerOutput = mixerIdentity(output)

        // Mixer - Conv kxk
        guard let convWeight = weights["\(prefix).mixer.conv_kxk.0.conv.weight"] else {
            return mixerOutput
        }

        let convBN = try loadBatchNorm(prefix: "\(prefix).mixer.conv_kxk.0.bn")
        let conv = Conv2d(weight: convWeight, padding: (1, 1))
        var convOut = conv(output)
        convOut = convBN(convOut)

        // Conv scale
        if let scaleWeight = weights["\(prefix).mixer.conv_scale.conv.weight"] {
            let scaleBN = try loadBatchNorm(prefix: "\(prefix).mixer.conv_scale.bn")
            let scaleConv = Conv2d(weight: scaleWeight, padding: (0, 0))
            let scale = scaleBN(scaleConv(convOut))
            convOut = convOut * sigmoid(scale)
        }

        mixerOutput = mixerOutput + convOut

        return mixerOutput
    }

    // MARK: - MLP

    private func mlp(_ x: MLXArray, prefix: String) throws -> MLXArray {
        // Conv
        guard let convWeight = weights["\(prefix).conv.conv.weight"] else {
            throw ModelError.weightNotFound("\(prefix).conv.conv.weight")
        }

        let convBN = try loadBatchNorm(prefix: "\(prefix).conv.bn")

        // Determine padding based on kernel size to preserve spatial dimensions
        // For kxk kernel, padding = k//2
        let kernelH = convWeight.shape[2]
        let kernelW = convWeight.shape[3]
        let padding = (kernelH / 2, kernelW / 2)

        let conv = Conv2d(weight: convWeight, padding: padding)
        var output = conv(x)
        output = convBN(output)

        // FC1
        guard let fc1Weight = weights["\(prefix).fc1.weight"],
              let fc1Bias = weights["\(prefix).fc1.bias"] else {
            throw ModelError.weightNotFound("\(prefix).fc1")
        }

        let fc1 = Linear(weight: fc1Weight, bias: fc1Bias)
        output = fc1(output)
        output = GELU()(output)

        // FC2
        guard let fc2Weight = weights["\(prefix).fc2.weight"],
              let fc2Bias = weights["\(prefix).fc2.bias"] else {
            throw ModelError.weightNotFound("\(prefix).fc2")
        }

        let fc2 = Linear(weight: fc2Weight, bias: fc2Bias)
        output = fc2(output)

        return output
    }

    // MARK: - Final Conv

    private func finalConv(_ x: MLXArray) throws -> MLXArray {
        let prefix = "module.visual.trunk.final_conv"

        // 1. Conv kxk (3x3 depthwise, 2048 -> 4096)
        guard let convWeight = weights["\(prefix).conv_kxk.0.conv.weight"] else {
            throw ModelError.weightNotFound("\(prefix).conv_kxk.0.conv.weight")
        }

        let convBN = try loadBatchNorm(prefix: "\(prefix).conv_kxk.0.bn")
        let conv = Conv2d(weight: convWeight, padding: (1, 1))
        var output = conv(x)
        output = convBN(output)
        output = ReLU()(output)

        // 2. Conv scale (1x1 conv for gating)
        guard let scaleWeight = weights["\(prefix).conv_scale.conv.weight"] else {
            throw ModelError.weightNotFound("\(prefix).conv_scale.conv.weight")
        }

        let scaleBN = try loadBatchNorm(prefix: "\(prefix).conv_scale.bn")
        let scaleConv = Conv2d(weight: scaleWeight, padding: (0, 0))
        let scale = scaleBN(scaleConv(output))
        output = output * sigmoid(scale)

        // 3. Squeeze-and-Excitation
        guard let fc1Weight = weights["\(prefix).se.fc1.weight"],
              let fc1Bias = weights["\(prefix).se.fc1.bias"],
              let fc2Weight = weights["\(prefix).se.fc2.weight"],
              let fc2Bias = weights["\(prefix).se.fc2.bias"] else {
            throw ModelError.weightNotFound("\(prefix).se")
        }

        // Global average pooling for SE
        let pooled = output.mean(axes: [1, 2], keepDims: true)

        // SE fc1 (reduction)
        let fc1 = Linear(weight: fc1Weight, bias: fc1Bias)
        var se = fc1(pooled)
        se = ReLU()(se)

        // SE fc2 (expansion)
        let fc2 = Linear(weight: fc2Weight, bias: fc2Bias)
        se = fc2(se)
        se = sigmoid(se)

        // Apply SE weights
        output = output * se

        return output
    }

    // MARK: - Head

    private func head(_ x: MLXArray) throws -> MLXArray {
        // Global Average Pooling
        let pooled = GlobalAvgPool2d()(x)

        // FC projection
        guard let fcWeight = weights["module.visual.trunk.head.fc.weight"] else {
            throw ModelError.weightNotFound("module.visual.trunk.head.fc.weight")
        }

        let fcBias = weights["module.visual.trunk.head.fc.bias"]

        let fc = Linear(weight: fcWeight, bias: fcBias)
        let output = fc(pooled)

        // L2 Normalization with safer epsilon
        let norm = output.square().sum(axis: -1, keepDims: true).sqrt()
        let normalized = output / (norm + 1e-6)

        return normalized
    }

    // MARK: - Helper Functions

    private func stagePrefix(_ stage: Int) -> String {
        return "module.visual.trunk.stages.\(stage)"
    }

    private func loadBatchNorm(prefix: String) throws -> BatchNorm2d {
        guard let weight = weights["\(prefix).weight"],
              let bias = weights["\(prefix).bias"],
              let mean = weights["\(prefix).running_mean"],
              let var_ = weights["\(prefix).running_var"] else {
            throw ModelError.weightNotFound("\(prefix).*")
        }

        return BatchNorm2d(
            weight: weight,
            bias: bias,
            runningMean: mean,
            runningVar: var_
        )
    }

    // MARK: - Transformer Block (Attention-based)

    private func transformerBlock(_ x: MLXArray, prefix: String) throws -> MLXArray {
        // Input: [batch, height, width, channels]

        // 1. LayerNorm + Attention
        guard let normWeight = weights["\(prefix).norm.weight"],
              let normBias = weights["\(prefix).norm.bias"] else {
            throw ModelError.weightNotFound("\(prefix).norm")
        }

        // Flatten spatial dimensions for attention: [batch, height, width, channels] -> [batch, height*width, channels]
        let batch = x.shape[0]
        let height = x.shape[1]
        let width = x.shape[2]
        let channels = x.shape[3]
        let seqLen = height * width

        var tokens = x.reshaped(batch, seqLen, channels)

        // LayerNorm
        let layerNorm = LayerNorm(weight: normWeight, bias: normBias)
        let normed = layerNorm(tokens)

        // Multi-head attention
        let attnOutput = try multiHeadAttention(normed, prefix: "\(prefix).token_mixer")

        tokens = tokens + attnOutput

        // 2. MLP
        let residual = tokens
        let mlpOutput = try mlp(tokens.reshaped(batch, height, width, channels), prefix: "\(prefix).mlp")
        let mlpFlattened = mlpOutput.reshaped(batch, seqLen, channels)

        tokens = residual + mlpFlattened

        // Reshape back to [batch, height, width, channels]
        let output = tokens.reshaped(batch, height, width, channels)

        return output
    }

    private func multiHeadAttention(_ x: MLXArray, prefix: String) throws -> MLXArray {
        // x: [batch, seq_len, channels]

        guard let qkvWeight = weights["\(prefix).qkv.weight"] else {
            throw ModelError.weightNotFound("\(prefix).qkv.weight")
        }

        let qkvBias = weights["\(prefix).qkv.bias"]

        guard let projWeight = weights["\(prefix).proj.weight"] else {
            throw ModelError.weightNotFound("\(prefix).proj.weight")
        }

        let projBias = weights["\(prefix).proj.bias"]

        let batch = x.shape[0]
        let seqLen = x.shape[1]
        let channels = x.shape[2]

        // QKV projection: [batch, seq_len, channels] -> [batch, seq_len, 3*channels]
        var qkv = matmul(x, qkvWeight.T)
        if let bias = qkvBias {
            qkv = qkv + bias
        }

        // Split into Q, K, V: each [batch, seq_len, channels]
        let qkvReshaped = qkv.reshaped(batch, seqLen, 3, channels)
        var q = qkvReshaped[0..., 0..., 0..<1, 0...].squeezed(axis: 2)
        var k = qkvReshaped[0..., 0..., 1..<2, 0...].squeezed(axis: 2)
        var v = qkvReshaped[0..., 0..., 2..<3, 0...].squeezed(axis: 2)

        // Reshape to [batch, 1, seq_len, channels] for MLX.scaledDotProductAttention
        // Treat as single-head attention (num_heads = 1)
        q = q.expandedDimensions(axis: 1)
        k = k.expandedDimensions(axis: 1)
        v = v.expandedDimensions(axis: 1)

        // Use MLX's optimized scaled dot-product attention (no causal mask for vision)
        let scale = 1.0 / sqrt(Float(channels))
        var output = scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: nil  // No masking for vision encoder
        )

        // Remove the num_heads dimension: [batch, 1, seq_len, channels] -> [batch, seq_len, channels]
        output = output.squeezed(axis: 1)

        // Output projection
        output = matmul(output, projWeight.T)
        if let bias = projBias {
            output = output + bias
        }

        return output
    }
}
