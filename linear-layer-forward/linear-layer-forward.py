def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    n_samples = len(X)
    d_in, d_out = len(W), len(W[0])

    y_out = []
    for i in range(n_samples):
        y_out_sample = []
        for k in range(d_out):
            sum = 0
            for j in range(d_in):
                sum += X[i][j] * W[j][k]
            y_out_sample.append(sum + b[k])
        y_out.append(y_out_sample)
    return y_out
                