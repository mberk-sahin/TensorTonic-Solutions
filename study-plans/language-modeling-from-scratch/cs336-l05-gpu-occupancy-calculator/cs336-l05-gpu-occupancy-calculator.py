def gpu_occupancy(
    threads_per_block: int, registers_per_thread: int, shared_mem_per_block: int,
    max_threads_per_sm: int, max_warps_per_sm: int, max_blocks_per_sm: int,
    max_registers_per_sm: int, max_shared_mem_per_sm: int, warp_size: int = 32,
) -> dict:
    """
    Returns a dict: blocks_per_sm (int), resident_warps (int), occupancy (float).
    """

    def ceil(num: int, den: int):
        return -(-num // den)

    def floor(num:int, den: int):
        return num // den

    # number of warps per block
    warps_per_block = ceil(threads_per_block, warp_size)

    # number of blocks allowed given max thread count allowed
    eff_threads_per_block = warps_per_block * warp_size
    num_block_max_thread = floor(max_threads_per_sm, eff_threads_per_block)

    # number of blocks allowed given max warp count allowed
    num_block_max_warp = floor(max_warps_per_sm, warps_per_block)

    resource_sets = [
        num_block_max_thread,
        num_block_max_warp,
        max_blocks_per_sm
    ]

    # number of blocks allowed given max register count allowed
    if registers_per_thread != 0:
        num_block_max_reg = floor(max_registers_per_sm, eff_threads_per_block * registers_per_thread) 
        resource_sets.append(num_block_max_reg)

    # number of blocks allowed given max shared memory allowed
    if shared_mem_per_block != 0:
        num_block_max_memory = floor(max_shared_mem_per_sm, shared_mem_per_block)
        resource_sets.append(num_block_max_memory)

    resident_blocks = min(resource_sets)
    resident_warps = resident_blocks * warps_per_block
    occupancy = resident_warps / max_warps_per_sm
    
    return {
        "blocks_per_sm": resident_blocks,
        "resident_warps": resident_warps,
        "occupancy": occupancy
    }