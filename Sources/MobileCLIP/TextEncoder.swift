import Foundation
import MLX
import MLXNN

/// MobileCLIP2-S4 Text Encoder
///
/// Architecture:
/// - Token Embedding (vocab_size=49408, embed_dim=768)
/// - Positional Embedding (context_length=77, embed_dim=768)
/// - 12 Transformer Layers
///   - LayerNorm → Multi-Head Attention → LayerNorm → MLP
///   - num_heads=12, head_dim=64
///   - MLP with 4x expansion (3072)
/// - Final LayerNorm
/// - Text Projection (768 → 768)
public class TextEncoder {

    private let weights: [String: MLXArray]
    private let numLayers: Int = 12
    private let embedDim: Int = 768
    private let numHeads: Int = 12
    private let headDim: Int = 64
    private let mlpDim: Int = 3072
    private let vocabSize: Int = 49408
    private let contextLength: Int = 77

    // Cache causal mask for better performance
    private let causalMask: MLXArray

    public init(weights: [String: MLXArray]) {
        self.weights = weights

        // Pre-compute causal mask once during initialization
        // Upper triangular matrix (0 for valid positions, 1 for masked)
        self.causalMask = Self.createCausalMask(seqLen: contextLength)
    }

    private static func createCausalMask(seqLen: Int) -> MLXArray {
        // Create upper triangular mask for causal attention using MLX's triu()
        // Position i can only attend to positions <= i
        // triu(k=1) creates upper triangular matrix with ones above diagonal
        return triu(MLXArray.ones([seqLen, seqLen]), k: 1)
    }

    // MARK: - Main Encode Function

    /// テキストを埋め込みベクトルにエンコード
    /// - Parameter tokens: [batch, context_length] のトークンID
    /// - Returns: [batch, embed_dim] の埋め込みベクトル
    public func encode(_ tokens: MLXArray) throws -> MLXArray {
        // 1. Token Embedding
        guard let tokenEmbeddingWeight = weights["module.text.token_embedding.weight"] else {
            throw ModelError.weightNotFound("module.text.token_embedding.weight")
        }
        var x = tokenEmbeddingWeight[tokens]

        // 2. Add Positional Embedding
        guard let positionalEmbedding = weights["module.text.positional_embedding"] else {
            throw ModelError.weightNotFound("module.text.positional_embedding")
        }
        x = x + positionalEmbedding

        // 3. Transformer Layers
        for layer in 0..<numLayers {
            x = try transformerBlock(x, layer: layer)
        }

        // 4. Final LayerNorm
        x = try layerNorm(x, prefix: "module.text.ln_final")

        // 5. Extract features at [EOS] token position
        // For CLIP, we use the features at the position of the highest token index
        // which corresponds to the [EOS] token
        // For simplicity, we'll use the last position
        // Shape: [batch, seq, embed_dim] -> [batch, embed_dim]
        let lastPosition = x.shape[1] - 1

        // Extract last token features for all batches at once using slicing
        x = x[0..., lastPosition, 0...]

        // 6. Text Projection
        guard let textProjection = weights["module.text.text_projection"] else {
            throw ModelError.weightNotFound("module.text.text_projection")
        }
        x = matmul(x, textProjection.T)

        // 7. L2 Normalization with safer epsilon
        let norm = x.square().sum(axis: -1, keepDims: true).sqrt()
        let normalized = x / (norm + 1e-6)

        // Evaluate the entire computation graph once at the end
        eval(normalized)

        return normalized
    }

    // MARK: - Transformer Block

    private func transformerBlock(_ x: MLXArray, layer: Int) throws -> MLXArray {
        let prefix = "module.text.transformer.resblocks.\(layer)"

        // 1. LayerNorm + Attention
        let attnInput = try layerNorm(x, prefix: "\(prefix).ln_1")
        let attnOutput = try attention(attnInput, prefix: "\(prefix).attn")
        var output = x + attnOutput

        // 2. LayerNorm + MLP
        let mlpInput = try layerNorm(output, prefix: "\(prefix).ln_2")
        let mlpOutput = try mlp(mlpInput, prefix: "\(prefix).mlp")
        output = output + mlpOutput

        // Do NOT call eval() here - let MLX optimize the computation graph
        return output
    }

    // MARK: - Multi-Head Attention

    private func attention(_ x: MLXArray, prefix: String) throws -> MLXArray {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]

        // 1. Input projection (Q, K, V)
        guard let inProjWeight = weights["\(prefix).in_proj_weight"],
              let inProjBias = weights["\(prefix).in_proj_bias"] else {
            throw ModelError.weightNotFound("\(prefix).in_proj")
        }

        // Linear projection: [batch, seq, embed_dim] @ [3*embed_dim, embed_dim]^T
        let qkv = matmul(x, inProjWeight.T) + inProjBias

        // Split into Q, K, V: [batch, seq, 3*embed_dim] -> 3 x [batch, seq, embed_dim]
        let qkvSplit = MLX.split(qkv, indices: [embedDim, embedDim * 2], axis: -1)
        var q = qkvSplit[0]
        var k = qkvSplit[1]
        var v = qkvSplit[2]

        // 2. Reshape for multi-head attention
        // [batch, seq, embed_dim] -> [batch, seq, num_heads, head_dim]
        q = q.reshaped(batchSize, seqLen, numHeads, headDim)
        k = k.reshaped(batchSize, seqLen, numHeads, headDim)
        v = v.reshaped(batchSize, seqLen, numHeads, headDim)

        // Transpose to [batch, num_heads, seq, head_dim]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // 3. Use MLX's optimized scaled dot-product attention with causal mask
        let scale = 1.0 / sqrt(Float(headDim))

        // Prepare causal mask for current sequence length
        // scaledDotProductAttention expects additive mask (0 for valid, -inf for masked)
        let mask = causalMask[0..<seqLen, 0..<seqLen]
        let maskValue: Float = -1e9
        let attentionMask = mask * maskValue  // Convert to additive mask

        var output = scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: attentionMask
        )

        // 7. Transpose back and reshape
        // [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads, head_dim]
        output = output.transposed(0, 2, 1, 3)

        // [batch, seq, num_heads, head_dim] -> [batch, seq, embed_dim]
        output = output.reshaped(batchSize, seqLen, embedDim)

        // 8. Output projection
        guard let outProjWeight = weights["\(prefix).out_proj.weight"],
              let outProjBias = weights["\(prefix).out_proj.bias"] else {
            throw ModelError.weightNotFound("\(prefix).out_proj")
        }

        output = matmul(output, outProjWeight.T) + outProjBias

        return output
    }

    // MARK: - MLP

    private func mlp(_ x: MLXArray, prefix: String) throws -> MLXArray {
        // c_fc: Linear expansion
        guard let fcWeight = weights["\(prefix).c_fc.weight"],
              let fcBias = weights["\(prefix).c_fc.bias"] else {
            throw ModelError.weightNotFound("\(prefix).c_fc")
        }

        var output = matmul(x, fcWeight.T) + fcBias
        output = gelu(output)

        // c_proj: Linear projection back
        guard let projWeight = weights["\(prefix).c_proj.weight"],
              let projBias = weights["\(prefix).c_proj.bias"] else {
            throw ModelError.weightNotFound("\(prefix).c_proj")
        }

        output = matmul(output, projWeight.T) + projBias

        return output
    }

    // MARK: - Helper Functions

    private func layerNorm(_ x: MLXArray, prefix: String) throws -> MLXArray {
        guard let weight = weights["\(prefix).weight"],
              let bias = weights["\(prefix).bias"] else {
            throw ModelError.weightNotFound("\(prefix)")
        }

        // Layer normalization along last dimension
        let eps: Float = 1e-5
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)

        let normalized = (x - mean) / sqrt(variance + eps)
        return normalized * weight + bias
    }

    private func gelu(_ x: MLXArray) -> MLXArray {
        // Use MLX's optimized GELU with fast approximation
        return geluFastApproximate(x)
    }

    private func softmax(_ x: MLXArray, axis: Int) -> MLXArray {
        let maxVal = MLX.max(x, axis: axis, keepDims: true)
        let exp_x = exp(x - maxVal)
        let sum_exp = MLX.sum(exp_x, axis: axis, keepDims: true)
        return exp_x / sum_exp
    }
}
