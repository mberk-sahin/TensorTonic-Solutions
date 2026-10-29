import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    # Write code here

    # compute offsets
    offset = size // 2
    kernel = []
    total = 0

    for i in range(size):
        sub_kernel = []
        for j in range(size):
            potential = ((i - offset) ** 2 + (j - offset) ** 2) / (2 * sigma ** 2)
            energy = math.exp(-potential)
            sub_kernel.append(energy)
            total += energy
        kernel.append(sub_kernel)

    # normalize the values
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total

    return kernel
            
            