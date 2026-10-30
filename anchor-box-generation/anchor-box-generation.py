def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    # Write code here

    anchor_list = []
    stride = image_size / feature_size

    # iterate over grids
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            # iterate over scales and aspect ratios
            for s in scales:
                for aratio in aspect_ratios:
                    w = s * aratio ** 0.5
                    h = s / aratio ** 0.5
                    anchor = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                    anchor_list.append(anchor)
    return anchor_list
                    