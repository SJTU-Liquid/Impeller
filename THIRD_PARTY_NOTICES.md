# Third-party code

Impeller's GPU implementation adapts code from [FluxZK](https://github.com/SJTU-Liquid/FluxZK), the artifact for the SC 2026 paper [*FluxZK: Scalable and Efficient Zero-Knowledge Proof Computation via GPU Acceleration*](https://sc26.conference-program.com/presentation/?id=pap593&sess=sess231). This includes portions of the field arithmetic, elliptic-curve, MSM, and NTT code under `core/src/`. FluxZK is licensed under MIT; its original copyright and license text is preserved in [FluxZK-MIT.txt](THIRD_PARTY_LICENSES/FluxZK-MIT.txt).

The Montgomery arithmetic implementation in `core/src/common/mont/src/field.cuh` also contains code attributed in that file to Supranational's [sppark](https://github.com/supranational/sppark), licensed under Apache-2.0. The Apache-2.0 license text is included in [Apache-2.0.txt](THIRD_PARTY_LICENSES/Apache-2.0.txt). The source-file copyright and SPDX notices remain in place.
