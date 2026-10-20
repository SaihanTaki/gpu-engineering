"""Theoretical occupancy of one SM, the same rules as the chapter 5 simulation.

Follows cudaOccMaxActiveBlocksPerMultiprocessor in cuda_occupancy.h (CUDA
Toolkit) with the default shared memory carveout and a kernel that has opted in
to the largest dynamic shared memory size. Limits per compute capability are
from the CUDA Programming Guide.
"""

import math

WARP = 32

# (major, minor): per-SM and per-block limits. Shared memory sizes are in bytes.
LIMITS = {
    (7, 5): dict(max_warps=32, max_blocks=16, regs_per_sm=65536, max_regs_per_block=65536,
                 max_regs_per_thread=255, smem_per_sm=64 * 1024, smem_per_block=64 * 1024,
                 smem_unit=256, reserved_smem=0),
    (8, 0): dict(max_warps=64, max_blocks=32, regs_per_sm=65536, max_regs_per_block=65536,
                 max_regs_per_thread=255, smem_per_sm=164 * 1024, smem_per_block=163 * 1024,
                 smem_unit=128, reserved_smem=1024),
    (9, 0): dict(max_warps=64, max_blocks=32, regs_per_sm=65536, max_regs_per_block=65536,
                 max_regs_per_thread=255, smem_per_sm=228 * 1024, smem_per_block=227 * 1024,
                 smem_unit=128, reserved_smem=1024),
}
REG_UNIT = 256
PARTITIONS = 4


def round_up(x, unit):
    return math.ceil(x / unit) * unit


def blocks_per_sm(cc, regs_per_thread, smem_per_block, block_size):
    """Returns (blocks, limits): resident blocks per SM and what each resource allows."""
    d = LIMITS[cc]
    warps_per_block = math.ceil(block_size / WARP)
    regs_per_warp = round_up(regs_per_thread * WARP, REG_UNIT)
    # The hardware check assumes a block's warps are spread over every partition.
    regs_assumed = regs_per_warp * round_up(warps_per_block, PARTITIONS)
    if regs_assumed > d["max_regs_per_block"] or regs_per_thread > d["max_regs_per_thread"]:
        regs_limit = 0
    elif regs_per_warp == 0:
        regs_limit = math.inf
    else:
        warps_per_partition = (d["regs_per_sm"] // PARTITIONS) // regs_per_warp
        regs_limit = warps_per_partition * PARTITIONS // warps_per_block

    smem = round_up(smem_per_block + d["reserved_smem"], d["smem_unit"])
    if smem > d["smem_per_block"] + d["reserved_smem"]:
        smem_limit = 0
    else:
        smem_limit = d["smem_per_sm"] // smem if smem else math.inf

    limits = {
        "warps": d["max_warps"] // warps_per_block if block_size <= 1024 else 0,
        "registers": regs_limit,
        "shared": smem_limit,
        "blocks": d["max_blocks"],
    }
    return min(limits.values()), limits


def theoretical_occupancy(cc, regs_per_thread, smem_per_block, block_size):
    """Active warps per SM divided by the maximum, 0 to 1."""
    blocks, _ = blocks_per_sm(cc, regs_per_thread, smem_per_block, block_size)
    return blocks * math.ceil(block_size / WARP) / LIMITS[cc]["max_warps"]
