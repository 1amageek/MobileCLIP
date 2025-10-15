#!/usr/bin/env python3
"""
Download MobileCLIP2 models from Hugging Face and convert to safetensors format for MLX-Swift.

This script downloads the PyTorch model from Hugging Face Hub and converts it
directly to the safetensors format supported by MLX-Swift.

Available MobileCLIP2 Models:
    - apple/MobileCLIP2-S0     (11.4M + 63.4M params, 71.5% ImageNet accuracy)
    - apple/MobileCLIP2-S2     (35.7M + 63.4M params, 77.2% ImageNet accuracy) [Recommended]
    - apple/MobileCLIP2-B      (86.3M + 63.4M params, 79.4% ImageNet accuracy)
    - apple/MobileCLIP2-S3     (125.1M + 123.6M params, 80.7% ImageNet accuracy)
    - apple/MobileCLIP2-L-14   (304.3M + 123.6M params, 81.9% ImageNet accuracy)
    - apple/MobileCLIP2-S4     (321.6M + 123.6M params, 81.9% ImageNet accuracy)

Usage:
    python convert_to_safetensors.py [--model MODEL_ID] [--output OUTPUT_DIR] [--filename FILENAME]

Examples:
    # Download recommended model (S2)
    python convert_to_safetensors.py

    # Download specific model
    python convert_to_safetensors.py --model apple/MobileCLIP2-B

    # Custom output location
    python convert_to_safetensors.py --output Sources/MobileCLIP2/Resources --filename model.safetensors
"""

import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file

def download_and_convert_from_huggingface(model_id: str, output_file: Path):
    """
    Download model from Hugging Face and convert to safetensors.

    Args:
        model_id: Hugging Face model ID (e.g., 'apple/MobileCLIP2-S2')
        output_file: Output safetensors file path
    """
    print(f"📥 Downloading model from Hugging Face: {model_id}")
    print()

    try:
        from transformers import AutoModel
    except ImportError:
        print("❌ Error: transformers package not found")
        print("   Install it with: pip install transformers")
        return False

    # Download model
    try:
        print("   Loading model...")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print()
        print("   Available MobileCLIP2 models:")
        print("     - apple/MobileCLIP2-S0")
        print("     - apple/MobileCLIP2-S2 [Recommended]")
        print("     - apple/MobileCLIP2-B")
        print("     - apple/MobileCLIP2-S3")
        print("     - apple/MobileCLIP2-L-14")
        print("     - apple/MobileCLIP2-S4")
        print()
        print(f"   Verify the model exists at: https://huggingface.co/{model_id}")
        return False

    # Get state dict
    print("   Extracting state dict...")
    state_dict = model.state_dict()

    print(f"   Total tensors: {len(state_dict)}")

    # Convert to safetensors format
    try:
        print(f"   Saving to: {output_file}")
        save_file(state_dict, str(output_file))
    except Exception as e:
        print(f"❌ Error saving safetensors file: {e}")
        return False

    print()
    print("✅ Conversion complete!")
    print(f"   Output: {output_file}")
    print(f"   Size: {output_file.stat().st_size / (1024**2):.2f} MB")

    return True

def verify_safetensors(output_file: Path):
    """Verify the generated safetensors file."""
    print("\n🔍 Verifying conversion...")
    from safetensors import safe_open

    with safe_open(output_file, framework="pt") as f:
        keys = list(f.keys())
        print(f"   Total keys in safetensors: {len(keys)}")

        # Show some sample keys
        print("   Sample keys:")
        for i, key in enumerate(keys[:10], 1):
            tensor = f.get_tensor(key)
            print(f"     {i}. {key}: {list(tensor.shape)}")

def main():
    parser = argparse.ArgumentParser(
        description="Download MobileCLIP from Hugging Face and convert to safetensors"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="apple/MobileCLIP2-S4",
        help="Hugging Face model ID (default: apple/MobileCLIP2-S4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Sources/MobileCLIP/Resources",
        help="Output directory (default: Sources/MobileCLIP/Resources)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="MobileCLIP2-S4.safetensors",
        help="Output filename (default: MobileCLIP2-S4.safetensors)",
    )

    args = parser.parse_args()

    # Define paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / args.filename

    # Download and convert
    success = download_and_convert_from_huggingface(args.model, output_file)

    if not success:
        return

    # Verify the conversion
    verify_safetensors(output_file)

    print("\n✅ All done!")
    print(f"   Model ready for MLX-Swift at: {output_file}")

if __name__ == "__main__":
    try:
        import safetensors
    except ImportError:
        print("❌ Error: safetensors package not found")
        print("   Install it with: pip install safetensors")
        exit(1)

    try:
        import torch
    except ImportError:
        print("❌ Error: torch package not found")
        print("   Install it with: pip install torch")
        exit(1)

    main()
