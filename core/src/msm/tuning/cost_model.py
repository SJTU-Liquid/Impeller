from typing import List
import math

threshold1 = 2 ** 17
threshold2 = 2 ** 20

def division_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b

# A batched MSM problem to optimize.
class Instance:
    # Field element bit width.
    lambd: int
    # Point element bit width.
    plambd: int
    # Bytes per point index.
    delta: int
    # Number of points is 2**k.
    k: int
    # MSM batch size; all MSMs share the same point set.
    n: int
    # GPU memory capacity.
    g: int

    # Initialize the class attributes.
    def __init__(self, lambd: int, plambd: int, delta: int, k: int, n: int, g: int):
        self.lambd = lambd  # Field element bit width.
        self.plambd = plambd
        self.delta = delta  # Bytes per point index.
        self.k = k  # Number of points is 2**k.
        self.n = n  # MSM batch size; the instances share one point set.
        self.g = g  # GPU memory capacity.

    def scaler_size(self) -> int:
        return division_ceil(self.lambd, 8)

    def point_size(self) -> int:
        return division_ceil(self.plambd, 8)

    def point_affine_size(self) -> int:
        return self.point_size() * 2

    def point_xyzz_size(self) -> int:
        return self.point_size() * 4


# Parameters to optimize.
class Parameters:
    # Window width in bits.
    s: int
    # Precomputation factor: precompute one point set for every alpha windows.
    alpha: int
    # Sub-MSM width.
    w: int
    # Sub-MSM height.
    h: int
    # Sub-MSM segment width.
    c: int

    def __init__(self, s: int, alpha: int, w: int, h: int, c: int):
        # Window width in bits.
        self.s = s
        # Precomputation factor: precompute one point set for every alpha windows.
        self.alpha = alpha
        # Sub-MSM width.
        self.w = w
        # Sub-MSM height.
        self.h = h
        # Sub-MSM segment width.
        self.c = c

    def n_weights(self) -> int:
        return 2 ** self.s


# Implementation parameters to estimate.
class Estimations:
    # Scatter stage time coefficient.
    r_scatter: float
    # Buckets-Sum stage time coefficient.
    r_buckets_sum: float
    # Buckets-Reduction stage time coefficient.
    r_buckets_reduction: float
    r_max1: float
    r_max2: float
    # Transfer time coefficient.
    r_transfer: float

    def __init__(self, r_scatter: float, r_buckets_sum: float, r_buckets_reduction: float, r_max1: float, r_max2: float, r_transfer: float):
        self.r_scatter = r_scatter  # Scatter stage time coefficient.
        self.r_buckets_sum = r_buckets_sum  # Buckets-Sum stage time coefficient.
        self.r_buckets_reduction = r_buckets_reduction  # Buckets-Reduce stage time coefficient.
        self.r_max1 = r_max1  # Scale factor 1 for the Scatter stage.
        self.r_max2 = r_max2  # Scale factor 2 for the Buckets-Sum stage.
        self.r_transfer = r_transfer  # Transfer time coefficient.


def n_total_windows(i: Instance, p: Parameters) -> int:
    return division_ceil(i.lambd, p.s)


def workload_scatter(i: Instance, p: Parameters) -> int:
    return p.c * n_total_windows(i, p)


def workload_buckets_sum(i: Instance, p: Parameters) -> int:
    return p.c * n_total_windows(i, p)


def workload_buckets_reduction(i: Instance, p: Parameters) -> int:
    return p.n_weights() * p.alpha


class Model:
    i: Instance
    p: Parameters
    e: Estimations

    def __init__(self, i: 'Instance', p: 'Parameters', e: 'Estimations'):
        self.i = i  # MSM instance data.
        self.p = p  # Algorithm parameters.
        self.e = e  # Time estimation coefficients.

    def s_points(self) -> int:
        """Space required for one precomputed point segment, in bytes."""
        return division_ceil(self.i.lambd, self.p.s * self.p.alpha) * self.p.c * self.i.point_affine_size()

    def t_points(self) -> float:
        return self.s_points() * self.e.r_transfer

    def s_scalers(self) -> int:
        """Space required for one scalar segment, in bytes."""
        return self.p.c * self.i.scaler_size()

    def t_scalers(self) -> int:
        return self.s_scalers() * self.e.r_transfer

    def s_points_chunk(self) -> int:
        return division_ceil(self.s_points(), self.p.h)

    def t_points_chunk(self) -> float:
        return self.t_points() / self.p.h

    def t_scatter(self) -> float:
        # Performance decreases when there are too few c values.
        if self.p.c < threshold1:
            return self.e.r_scatter * workload_scatter(self.i, self.p) * (self.e.r_max1 ** math.log2(threshold1 / self.p.c))
        return self.e.r_scatter * workload_scatter(self.i, self.p)

    def s_indices(self) -> int:
        """Space required for point indices produced by Scatter, in bytes."""
        return self.p.c * n_total_windows(self.i, self.p) * self.i.delta

    def t_buckets_sum(self) -> float:
        # Performance decreases when there are too few c values.
        if self.p.c < threshold2:
            return self.e.r_buckets_sum * workload_buckets_sum(self.i, self.p) * (
                        self.e.r_max2 ** math.log2(threshold2 / self.p.c))
        return self.e.r_buckets_sum * workload_buckets_sum(self.i, self.p)

    def s_buckets(self) -> int:
        """Space required for buckets produced by Buckets-Sum, in bytes."""
        return self.p.n_weights() * self.p.alpha * self.i.point_xyzz_size() / 2

    def t_buckets_reduction(self) -> float:
        return self.e.r_buckets_reduction * workload_buckets_reduction(self.i, self.p)

    def n_rows(self) -> int:
        return division_ceil(self.i.n, self.p.h)

    def n_cols(self) -> int:
        return division_ceil(2 ** self.i.k, self.p.w * self.p.c)


def total_time(m: Model) -> float:
    n_cols = m.n_cols()
    n_rows = m.n_rows()

    t_first = m.t_scalers() + max(m.t_scatter(), m.t_points())
    t_inner = max(m.p.h * m.t_scalers() + m.t_points(), m.p.h * (m.t_scatter() + m.t_buckets_sum()))
    t_last = m.p.h * m.t_buckets_sum() + m.p.h * m.t_buckets_reduction() + (m.p.h - 1) * m.t_scatter()
    t_sub = t_first + (m.p.w - 1) * t_inner + t_last

    # print(f"alpha:{m.p.alpha} s:{m.p.s} w:{m.p.w} c:{m.p.c} h:{m.p.h}")
    # print(f"{m.t_scalers()} {m.t_points()} {m.t_scatter()} {m.t_buckets_sum()} {m.t_buckets_reduction()}")
    # print(f"{t_first} {t_inner} {t_last}")
    # print(f"{n_rows * n_cols * t_sub}")

    return n_rows * n_cols * t_sub

def memory(m: Model) -> float:
    """Peak GPU memory usage for the entire MSM batch, in bytes."""
    return m.s_points() * 4 + m.s_scalers() * 4 + m.s_indices() * 4 + m.p.h * m.s_buckets() * 4

def constrain(m: Model) -> bool:
    return memory(m) < m.i.g

import itertools

def grid(fs, *args):
    accu = [()]
    for i, f in enumerate(fs[0:]):
        accu = itertools.chain(*(
            [(*old, new) for new in f(old, *args[i])]
            for old in accu
        ))
    return accu

def ss(old, plambd):
    if plambd < 400:
        return range(8, 24)
    else:
        return range(16, 31)

def alphas(old, lambd):
    s, = old
    return range(1, division_ceil(lambd, s) + 1)

def ecs(old, k, plambd):
    if k < 18:
        return range(max(k-4, 1), k + 1)
    elif k <= 25:
        return range(18, k + 1)
    else:
        if plambd < 400:
            return range(22, k + 1)
        else:
            return range(20, k + 1)

def hs(old, n):
    if n == 1:
        result = [1]
        return result
    result = []
    power = 1
    while power <= n:
        power *= 2
        result.append(power)
    return result

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=20, type=int, help="degree of the MSM")
    parser.add_argument("--n", default=1, type=int, help="number of MSMs")
    parser.add_argument("--l", default=256, type=int, help="bits of the scalar")
    parser.add_argument("--p", type=int, default=256, help="bits of the point base field")
    parser.add_argument("--mem", default=40 * 2 ** 30, type=int, help="memory limit")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    k = args.k
    n = args.n
    l = args.l
    p = args.p
    gpu_mem = args.mem

    # k = 20
    # n = 1
    # l = 256
    # p = 256


    # k = 26
    # n = 16
    # l = 768
    # p = 768

    search_grid = grid([ss, alphas, ecs, hs], [p], [l], [k, p], [n])

    models = [
        Model(
            i=Instance(
                lambd=l,
                plambd=p,
                delta=8,
                k=k,
                n=n,
                g=gpu_mem
            ),
            p=Parameters(
                s=s,
                alpha=alpha,
                w=2 ** (k - ec),
                h=h,
                c=2 ** ec
            ),
            e=Estimations(
                r_scatter=1,
                r_buckets_sum=7,
                r_buckets_reduction=40,
                r_max1=1.5,
                r_max2=1.05,
                r_transfer=0.8
            )
        )
        for s, alpha, ec, h in search_grid
    ]

    best = min((m for m in models if constrain(m)), key=lambda m: total_time(m))

    print(best.p.alpha)
    print(best.p.s)
    print(best.p.c)
    print(best.p.w)
    print(best.p.h)
    print('\n')

# A100
# bn254 255
# bls12381 381,256
# mnt4753 768,768
# r_scatter=1,
# r_buckets_sum=7,
# r_buckets_reduction=40,
# r_max1=1.5,
# r_max2=1.05,
# r_transfer=0.8
