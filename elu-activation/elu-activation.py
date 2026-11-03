import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """

    out = []
    # Write code here
    for i in range(len(x)):
        x_i = x[i]
        if x_i > 0:
            h_i = x_i
        else:
            h_i = alpha * (math.exp(x_i) - 1)
        out.append(h_i)

    return out