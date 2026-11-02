# Disorder / Anderson

[All models](../../README.md#browse-by-model) · [Clean PA](preferential_attachment.md)

All these operators have hopping one and uniform diagonal disorder,
H(a)=A+a diag(ξ), ξᵢ iid Uniform[−1,1). The experiments differ in graph grids,
amplitudes, windows, and replication; they must not be pooled as one run.

| Experiment | Graphs / operators | Amplitude convention | Detailed page |
|---|---|---|---|
| Final affine-PA sweep | 14 graphs, 112 operators | Raw a=0,0.5,1,2,4,8,16,32 | [PA diagonal disorder](../experiments/pa_diagonal_disorder.md) |
| Replicated PA extension | 27 graphs, 189 operators | a/√20=0,0.5,2,8 | [Anderson comparison](../experiments/anderson_benchmark.md) |
| Random 3-regular benchmark | 9 graphs, 99 operators including a=9 extension | Raw a=0,4,8,9,10,16; W=2a | [Anderson comparison](../experiments/anderson_benchmark.md) |

The 14-graph sweep pairs one potential direction across amplitudes and has no
graph replication within a condition. The replicated extension nests two
potential streams within each of three graphs per model and size. Its
uncertainty unit is the graph. Random regular graphs provide a separate
benchmark, not a replacement or a rewiring of PA graphs.
