// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MobileCLIP",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "MobileCLIP",
            targets: ["MobileCLIP"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", branch: "main")
    ],
    targets: [
        .target(
            name: "MobileCLIP",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples")
            ],
            resources: [
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "MobileCLIPTests",
            dependencies: ["MobileCLIP"]
        ),
    ]
)
