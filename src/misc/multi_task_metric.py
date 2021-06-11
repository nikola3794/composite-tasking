def performance_drop_metric(results_st, results_mt):
    """
    Calculates the performance drop of the multi-tasking system
    compared to a single-tasking baseline.
    Hardcoded for PASCAL-MT 5 tasks.
    """
    performance_drop = 0.0
    for k in results_st:
        l_k = 1.0 if k=="normals" else 0.0
        performance_drop += (-1.0)**l_k * (results_mt[k] - results_st[k]) / results_st[k]
    performance_drop /= len(results_st)
    performance_drop *= 100

    print(performance_drop)
    return performance_drop

if __name__ == "__main__":
    st_res = {
        "edges": 68.6,
        "seg": 62.45,
        "parts": 52.59,
        "normals": 16.93,
        "saliency": 67.81,
        
    }
    exp_res = {
        "edges": 68.5,
        "seg": 62.42,
        "parts": 52.50,
        "normals": 17.39,
        "saliency": 67.89,
    }

    performance_drop = performance_drop_metric(
        results_st=st_res,
        results_mt=exp_res
    )

    

    