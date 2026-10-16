"""Map and clusters of windows: standardise the chosen features, PCA, UMAP to a 2-D map, HDBSCAN.

Missing feature values (e.g. no inter-click interval) take the feature's median. PCA keeps the components
that explain ``pca_variance`` of the variance (or ``pca_components`` of them). UMAP makes the 2-D map; the
clusters are found by HDBSCAN on the map itself (``cluster_dims`` = 2: what you see is what is clustered) or
on a separate UMAP with more dimensions (``cluster_dims`` > 2: often cleaner clusters, but they may look
split or mixed on the map). Points HDBSCAN can't place are unclustered (-1).
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ClusterParams:
    """Clustering settings.

    pca_variance        share of variance the PCA components keep (0-1)
    pca_components      number of PCA components instead (0 = use pca_variance)
    n_neighbors         UMAP neighbours: larger = more global structure, smaller = more local detail
    min_dist            UMAP minimum distance between points on the map (0 = tight clumps)
    metric              UMAP distance ("euclidean", "cosine", "manhattan")
    cluster_dims        dimensions HDBSCAN clusters in (2 = the map)
    min_cluster_size    HDBSCAN smallest cluster (windows)
    min_samples         HDBSCAN neighbourhood density (0 = same as min_cluster_size): larger = more unclustered
    selection           HDBSCAN "eom" (fewer, larger clusters) or "leaf" (more, finer clusters)
    seed                random seed (UMAP)
    """

    pca_variance: float = 0.95
    pca_components: int = 0
    n_neighbors: int = 15
    min_dist: float = 0.05
    metric: str = "euclidean"
    cluster_dims: int = 2
    min_cluster_size: int = 25
    min_samples: int = 0
    selection: str = "leaf"
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def feature_matrix(windows, cols):
    """Standardised features (missing values: the column median; constant columns dropped)."""
    X = windows[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    sd = X.std(axis=0)
    keep = sd > 0
    return (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep], [c for c, k in zip(cols, keep) if k]


def _umap(X, n, params, min_dist):
    import sys
    if "umap" not in sys.modules:          # don't let umap import TensorFlow (for ParametricUMAP, unused): slow
        sys.modules.setdefault("umap.parametric_umap", None)
    from umap import UMAP
    return UMAP(n_components=n, n_neighbors=min(params.n_neighbors, len(X) - 1), min_dist=min_dist,
                metric=params.metric, random_state=params.seed, n_jobs=1).fit_transform(X)


def embed_and_cluster(windows, cols, params: ClusterParams = None, cache=None):
    """windows (one row per point, feature columns ``cols``) -> (DataFrame ``x, y, cluster`` in the points'
    order, info dict). Clusters are numbered 0.. by size (largest first); -1 = unclustered.

    ``cache``: a file (.npz) keeping the PCA + UMAP result; when the features and the PCA/UMAP settings are
    unchanged, only HDBSCAN runs again (seconds instead of minutes)."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    params = params or ClusterParams()
    X, used = feature_matrix(windows, cols)
    n = len(X)
    if n < 5 or X.shape[1] == 0:
        raise ValueError(f"Too few windows ({n}) or no varying features to cluster")
    nc = params.pca_components or params.pca_variance
    if isinstance(nc, float) and nc >= 1:
        nc = None
    if isinstance(nc, int):
        nc = min(nc, X.shape[1], n)
    from speciesclassifiers.progress import step
    import hashlib
    key = hashlib.sha1(np.ascontiguousarray(X).tobytes() + repr(
        (nc, params.n_neighbors, params.min_dist, params.metric, params.seed, params.cluster_dims)).encode()).hexdigest()
    hit = None
    if cache is not None and Path(cache).is_file():
        try:
            z = np.load(cache)
            if str(z["key"]) == key:
                hit = z
        except Exception:      # noqa: BLE001  (unreadable cache: recompute)
            hit = None
    if hit is not None:
        xy, space, ncomp, var = hit["xy"], hit["space"], int(hit["ncomp"]), float(hit["var"])
        print(f"  {n} points, {len(used)} features: map unchanged (same features and map settings); "
              f"clustering again")
    else:
        step("PCA")
        pca = PCA(n_components=nc, svd_solver="full", random_state=params.seed).fit(X)
        Z = pca.transform(X)
        ncomp, var = int(Z.shape[1]), float(pca.explained_variance_ratio_.sum())
        print(f"  {n} points, {len(used)} features -> {ncomp} PCA components ({var:.0%} of the variance)")
        print(f"  UMAP map (neighbours {params.n_neighbors}, min distance {params.min_dist})...")
        step(f"UMAP map of {n} points (no count available; can take minutes for many points)")
        xy = _umap(Z, 2, params, params.min_dist)
        space = xy
        if params.cluster_dims and params.cluster_dims > 2:
            print(f"  UMAP to {params.cluster_dims} dimensions for clustering...")
            step(f"UMAP to {params.cluster_dims} dimensions")
            space = _umap(Z, params.cluster_dims, params, 0.0)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, key=key, xy=xy, space=space, ncomp=ncomp, var=var)
    step("HDBSCAN clusters")
    mcs = max(2, min(params.min_cluster_size, n))
    cl = HDBSCAN(min_cluster_size=mcs, min_samples=params.min_samples or None,
                 cluster_selection_method=params.selection, copy=True).fit_predict(space)
    order = pd.Series(cl[cl >= 0]).value_counts().index
    remap = {old: new for new, old in enumerate(order)}
    cl = np.array([remap.get(v, -1) for v in cl])
    print(f"  HDBSCAN ({params.selection}, smallest {mcs}, density {params.min_samples or mcs}): "
          f"{len(order)} clusters; {np.mean(cl < 0):.0%} of points unclustered")
    out = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cluster": cl})
    info = {"n_windows": n, "features": used, "pca_components": ncomp,
            "pca_variance": var, "n_clusters": len(order),
            "unclustered": float(np.mean(cl < 0)), "params": params.to_dict()}
    return out, info
