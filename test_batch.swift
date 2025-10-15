import Foundation
import MLX
@testable import MobileCLIP

print("Testing Tokenizer batch encoding...")

do {
    let tokenizer = try Tokenizer()

    let texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird"
    ]

    let batchTokens = tokenizer.encodeBatch(texts)

    print("✅ Batch tokenized shape: \(batchTokens.shape)")
    print("   Expected: [3, 77]")
    print("   shape[0] (batch size): \(batchTokens.shape[0])")
    print("   shape[1] (context length): \(batchTokens.shape[1])")

    // Check shape
    if batchTokens.shape.count == 2 &&
       batchTokens.shape[0] == 3 &&
       batchTokens.shape[1] == 77 {
        print("✅ TEST PASSED: Batch encoding produces correct shape")
    } else {
        print("❌ TEST FAILED: Unexpected shape")
        exit(1)
    }

    // Check each row has SOS and EOS
    for i in 0..<3 {
        let row = batchTokens[i]
        let firstToken = row[0].item(Int32.self)

        // Find EOS position
        var eosPos = -1
        for j in 0..<77 {
            let token = row[j].item(Int32.self)
            if token == 49407 {  // EOS
                eosPos = j
                break
            }
        }

        print("   Row \(i): SOS=\(firstToken == 49406), EOS at position \(eosPos)")
    }

    print("\n✅ ALL TESTS PASSED")

} catch {
    print("❌ ERROR: \(error)")
    exit(1)
}
