# Impeller

Impeller is the compiler and runtime for graph-based zero-knowledge proof computation on heterogeneous CPU-GPU systems. The Rust crate names retain the earlier `zkpoly` prefix.

## Papers

- **Impeller: Scalable and Efficient Zero-Knowledge Proof Computation Framework via Graph Compilation.** Liukun Yu<sup>*</sup>, Xinwei Qiang<sup>*</sup>, Shixuan Sun, Chen Chen, Jingwen Leng, and Jin Dong. ASPLOS 2027. This repository contains the core compiler, runtime, and GPU code for the paper. See the [author's publication listing](https://tie-pilot-qxw.github.io/) while the proceedings link is pending.
- **[FluxZK: Scalable and Efficient Zero-Knowledge Proof Computation via GPU Acceleration](https://sc26.conference-program.com/presentation/?id=pap593&sess=sess231).** Xinwei Qiang<sup>*</sup>, Liukun Yu<sup>*</sup>, Xiyu Wang, Zhengyi Li, Shixuan Sun, Jingwen Leng, Chen Chen, Jiaping Gui, Zhenzhe Zheng, Jin Dong, and Minyi Guo. SC 2026. Several Impeller GPU kernels are adapted from the [FluxZK repository](https://github.com/SJTU-Liquid/FluxZK); see [third-party notices](THIRD_PARTY_NOTICES.md).

<sup>*</sup> Equal contribution.

The companion [Halo2 integration](https://github.com/SJTU-Liquid/halo2) and [zkEVM evaluation circuits](https://github.com/SJTU-Liquid/zkevm-circuits) reproduce the paper's proving workloads. They are separate repositories because the latter also depends on a pinned `snark-verifier` and `halo2-lib` pair.

This release is based on `danzou1ge6/zkpoly-compiler` commit `45f3e15936d8bf474c9e56946f4f8b92626ff153` (`artifect-evaluation`).

## Overview

Zero-knowledge proofs (ZKPs) are cryptographic protocols that allow one party to prove to another that a statement is true without revealing any information beyond the validity of the statement itself. These proofs are fundamental to many blockchain and cryptographic applications, including privacy-preserving transactions, confidential computing, and verifiable computation.

Impeller addresses the computational challenges of ZKPs by:

1. Providing high-level abstractions for polynomial operations
2. Implementing efficient GPU accelerated cryptographic primitives
3. Optimizing memory usage and execution across heterogeneous devices
4. Automating the compilation and optimization of proof circuits

## System Architecture

Impeller is structured as a multi-layered system:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    User Code (ZKP Application)              │
  └───────────────────────────────┬─────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────┐
 ┌┤                         Compiler                            │
 |├─────────────────┬─────────────────────────┬─────────────────┤
 |│      AST        │        Type2 IR         │     Type3 IR    │
 |│  (Front-end)    │   (Middle-end + Opts)   │  (Back-end)     │
 |└─────────────────┴───────────────┬─────────┴─────────────────┘
 |                                  │
 |┌─────────────────────────────────▼─────────────────────────────┐
 |│                           Runtime                             ├─┐
 |├──────────────────┬───────────────────────┬────────────────────┤ |
 |│  Instruction Set │   Memory Management   │ Device Coordination│ |
 |└──────────────────┴──────────┬────────────┴────────────────────┘ |
 |                              │                                   |
 |┌─────────────────────────────▼───────────────────────────────┐   |
 └>                        Core Libraries                       │   |
  ├───────────────┬───────────────┬─────────────┬───────────────┤   |
  │   NTT Ops     │   Polynomial  │     MSM     │  Fused Kernels│   |
  │               │      Ops      │             │               │   |
  └───────────────┴───────────────┴─────────────┴───────────────┘   |
                                                                    |
  ┌──────────────────────────────────────────────────────────────┐  |
  │                    Low-level Infrastructure                  <──┘
  ├────────────────────────────┬─────────────────────────────────┤
  │        CUDA API            │        Memory Pool              │
  └────────────────────────────┴─────────────────────────────────┘
```

## Key Components

### Compiler

The compiler takes high-level ZKP specifications and progressively lowers them through three intermediate representations (IRs):

1. **AST (Abstract Syntax Tree)**: Front-end representation that preserves the structure of user code.
   - Type system for polynomials (coefficient and Lagrange forms), scalars, points, arrays, and tuples
   - Expression representation for arithmetic operations
   - Common cryptographic primitives like MSM and NTT

2. **Type2 IR**: Mid-level representation with explicit typing and memory considerations.
   - Computation graph representation for optimizations
   - Precise type system with memory size information
   - Device targeting (CPU/GPU)
   - Multiple optimization passes including memory planning, kernel fusion, and graph scheduling

3. **Type3 IR**: Low-level representation focused on execution.
   - Instruction sequence organized into execution tracks
   - Explicit memory addressing and management
   - Stream-based execution model
   - Device-specific optimization

### Runtime

The runtime is responsible for executing the compiled programs efficiently:

- **Instruction Set**: Fine-grained control over memory, computation, and synchronization
- **Cross-device Execution**: Seamless operation across CPU and GPU
- **Memory Management**: Efficient allocation and deallocation across devices
- **Synchronization**: Event-based coordination between operations
- **Parallel Execution**: Multi-threaded operations and stream-based concurrency

### Core Libraries

The core module implements fundamental cryptographic and mathematical operations:

- **NTT Operations**: Fast Number Theoretic Transform for polynomial operations
- **Polynomial Operations**: Addition, multiplication, evaluation, division, and other operations
- **MSM (Multi-Scalar Multiplication)**: Efficient batched operations for elliptic curve points
- **Fused Kernels**: Dynamically generated operations for optimal performance

### Support Libraries

- **CUDA API**: Safe Rust interface to NVIDIA's CUDA runtime API
- **Memory Pool**: High-performance slab allocator for CUDA pinned memory
- **Common Utilities**: Shared data structures and algorithms used throughout the system

## Common Operations

Impeller supports common operations required in zero-knowledge proofs:

- **Polynomial Arithmetic**: Addition, subtraction, multiplication in both coefficient and Lagrange forms
- **NTT Transforms**: Converting between coefficient and Lagrange representations
- **Polynomial Evaluation**: Evaluating polynomials at specific points
- **Kate Polynomial Division**: Computing quotients for polynomial commitments
- **Multi-Scalar Multiplication**: Efficient implementation of ∑(scalar_i · point_i)
- **Batch Operations**: Optimized parallel versions of common operations
- **Fiat-Shamir Transformations**: Transcript management for non-interactive proofs

## Environment Setup

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (the paper evaluation used CUDA 12.8)
- Rust nightly `2025-09-20` (pinned in `rust-toolchain.toml`)
- [`xmake`](https://xmake.io/) for CUDA components

### Environment Variables

```
export CUDA_PATH=/path/to/cuda  # Tell Cargo where CUDA is installed

export ZKPOLY_COMPILER_PROJECT_ROOT="$(pwd)" # Run this from the Impeller repository root
```

The optional `IMPELLER_SECONDARY_TMPDIR` environment variable selects a second disk for the runtime transfer benchmark; its default is the system temporary directory.

### Building

```bash
# Clone the repository
git clone https://github.com/SJTU-Liquid/Impeller.git
cd Impeller

# Build the compiler crate
cargo build --release -p zkpoly_compiler
```

For proof generation workloads, use the companion [Halo2 integration](https://github.com/SJTU-Liquid/halo2) and [zkEVM circuits](https://github.com/SJTU-Liquid/zkevm-circuits). The Halo2 repository pins this compiler as a submodule. Set `ZKPOLY_COMPILER_PROJECT_ROOT` to the compiler checkout before running its GPU proof path.

## Performance Considerations

- **Memory Transfer**: Data transfer between CPU and GPU can be a bottleneck; minimize when possible
- **Kernel Fusion**: The compiler automatically fuses compatible operations to reduce kernel launches
- **Memory Planning**: Efficient memory allocation and reuse is critical for performance
- **Device Selection**: Operations are automatically assigned to the most appropriate device
- **Stream Management**: Use concurrent streams for overlapped execution

## License

Original Impeller code is available under the [MIT License](LICENSE). Reused code retains its own copyright and license notices; see [third-party notices](THIRD_PARTY_NOTICES.md).
