import Foundation
import MLX

/// CLIP BPE Tokenizer
///
/// Based on OpenAI CLIP's simple_tokenizer.py
/// Uses byte-level BPE encoding with vocabulary from bpe_simple_vocab_16e6.txt
public class Tokenizer {

    // MARK: - Constants

    private let contextLength: Int = 77
    private let vocabSize: Int = 49408

    // Special tokens
    private let sosToken: Int = 49406  // Start of sequence
    private let eosToken: Int = 49407  // End of sequence
    private let padToken: Int = 0      // Padding

    // MARK: - Properties

    private let byteEncoder: [UInt8: String]
    private let byteDecoder: [String: UInt8]
    private let encoder: [String: Int]
    private let decoder: [Int: String]
    private let bpeRanks: [Pair: Int]

    // MARK: - Initialization

    public init() throws {
        // Initialize byte encoder/decoder
        self.byteEncoder = Self.bytesToUnicode()
        self.byteDecoder = Dictionary(uniqueKeysWithValues: byteEncoder.map { ($1, $0) })

        // Load BPE merges from bundle
        guard let url = Bundle.module.url(forResource: "bpe_simple_vocab_16e6", withExtension: "txt", subdirectory: "Resources") else {
            throw TokenizerError.resourceNotFound("bpe_simple_vocab_16e6.txt")
        }

        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)

        // Skip version line
        let merges = lines.dropFirst().filter { !$0.isEmpty }

        // Build BPE ranks
        var bpeRanks = [Pair: Int]()
        for (i, line) in merges.enumerated() {
            let parts = line.split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                let pair = Pair(String(parts[0]), String(parts[1]))
                bpeRanks[pair] = i
            }
        }
        self.bpeRanks = bpeRanks

        // Build vocabulary
        // Base vocabulary: individual bytes
        var vocab = [String]()
        for byte in 0..<256 {
            if let char = byteEncoder[UInt8(byte)] {
                vocab.append(char)
            }
        }

        // Add merged tokens (merges create new vocabulary entries)
        for pair in bpeRanks.keys.sorted(by: { bpeRanks[$0]! < bpeRanks[$1]! }) {
            vocab.append(pair.first + pair.second)
        }

        // Add special tokens
        vocab.append("<|startoftext|>")  // 49406
        vocab.append("<|endoftext|>")    // 49407

        // Build encoder/decoder dictionaries
        self.encoder = Dictionary(uniqueKeysWithValues: vocab.enumerated().map { ($1, $0) })
        self.decoder = Dictionary(uniqueKeysWithValues: vocab.enumerated().map { ($0, $1) })
    }

    // MARK: - Encoding

    /// Encode text to token IDs
    /// - Parameter text: Input text
    /// - Returns: MLXArray of shape [1, contextLength]
    public func encode(_ text: String) -> MLXArray {
        // Tokenize with BPE
        let tokens = bpeEncode(text)

        // Convert to IDs
        var tokenIds = [Int32]()
        tokenIds.append(Int32(sosToken))  // Start token

        for token in tokens.prefix(contextLength - 2) {
            if let id = encoder[token] {
                tokenIds.append(Int32(id))
            }
        }

        tokenIds.append(Int32(eosToken))  // End token

        // Pad to context length
        while tokenIds.count < contextLength {
            tokenIds.append(Int32(padToken))
        }

        // Truncate if too long (ensure EOS is at the end)
        if tokenIds.count > contextLength {
            tokenIds = Array(tokenIds.prefix(contextLength - 1)) + [Int32(eosToken)]
        }

        // Convert to MLXArray [1, contextLength]
        let array = MLXArray(tokenIds)
        return array.reshaped(1, contextLength)
    }

    /// Encode multiple texts in batch
    /// - Parameter texts: Array of input texts
    /// - Returns: MLXArray of shape [batchSize, contextLength]
    public func encodeBatch(_ texts: [String]) -> MLXArray {
        let encodedTexts = texts.map { encode($0) }

        // Stack into batch and squeeze extra dimension
        // encode() returns [1, 77], stacking gives [batchSize, 1, 77]
        // squeeze to get [batchSize, 77]
        return MLX.stacked(encodedTexts, axis: 0).squeezed(axis: 1)
    }

    // MARK: - Decoding

    /// Decode token IDs to text
    /// - Parameter tokens: Token IDs
    /// - Returns: Decoded text
    public func decode(_ tokens: MLXArray) -> String {
        let tokenIds = tokens.asArray(Int32.self)

        var textTokens = [String]()
        for id in tokenIds {
            let intId = Int(id)
            if intId == sosToken || intId == eosToken || intId == padToken {
                continue
            }
            if let token = decoder[intId] {
                textTokens.append(token)
            }
        }

        // Join and decode bytes
        let text = textTokens.joined()
        let bytes = text.compactMap { byteDecoder[String($0)] }

        if let decodedText = String(bytes: bytes, encoding: .utf8) {
            return decodedText
        }

        return text
    }

    // MARK: - BPE Algorithm

    private func bpeEncode(_ text: String) -> [String] {
        // Lowercase and clean text
        let cleanedText = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        // Split into words (whitespace and punctuation)
        let pattern = #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return []
        }

        let nsText = cleanedText as NSString
        let matches = regex.matches(in: cleanedText, range: NSRange(location: 0, length: nsText.length))

        var tokens = [String]()
        for match in matches {
            let word = nsText.substring(with: match.range)

            // Convert word to bytes and then to unicode
            let wordBytes = Array(word.utf8)
            let wordTokens = wordBytes.map { byteEncoder[$0] ?? "" }

            // Apply BPE to word
            let bpeTokens = bpe(wordTokens)
            tokens.append(contentsOf: bpeTokens)
        }

        return tokens
    }

    private func bpe(_ tokens: [String]) -> [String] {
        if tokens.count <= 1 {
            return tokens
        }

        var word = tokens

        while true {
            // Find the pair with lowest rank
            var minPair: Pair? = nil
            var minRank = Int.max

            for i in 0..<(word.count - 1) {
                let pair = Pair(word[i], word[i + 1])
                if let rank = bpeRanks[pair], rank < minRank {
                    minPair = pair
                    minRank = rank
                }
            }

            guard let pair = minPair else {
                break
            }

            // Merge the pair
            var newWord = [String]()
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == pair.first && word[i + 1] == pair.second {
                    newWord.append(pair.first + pair.second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }

            word = newWord

            if word.count == 1 {
                break
            }
        }

        return word
    }

    // MARK: - Helper Functions

    /// Create byte to unicode mapping
    /// Avoids mapping to whitespace/control characters
    private static func bytesToUnicode() -> [UInt8: String] {
        // Printable ASCII range
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var chars = bytes

        var n = 0
        for b in 0..<256 {
            if !bytes.contains(b) {
                bytes.append(b)
                chars.append(256 + n)
                n += 1
            }
        }

        var mapping = [UInt8: String]()
        for (byte, char) in zip(bytes, chars) {
            if let scalar = UnicodeScalar(char) {
                mapping[UInt8(byte)] = String(Character(scalar))
            }
        }

        return mapping
    }
}

// MARK: - Supporting Types

/// Pair of tokens for BPE merging
private struct Pair: Hashable {
    let first: String
    let second: String

    init(_ first: String, _ second: String) {
        self.first = first
        self.second = second
    }
}

/// Tokenizer errors
public enum TokenizerError: Error, LocalizedError {
    case resourceNotFound(String)
    case encodingError(String)

    public var errorDescription: String? {
        switch self {
        case .resourceNotFound(let resource):
            return "Tokenizer resource not found: \(resource)"
        case .encodingError(let message):
            return "Encoding error: \(message)"
        }
    }
}
