def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    set_reco = set(recommended[:k])
    set_rel = set(relevant)
    inter_reco_rel = set_reco.intersection(set_rel)

    precision = len(inter_reco_rel) / k
    recall = len(inter_reco_rel) / len(relevant)
    return [precision, recall]