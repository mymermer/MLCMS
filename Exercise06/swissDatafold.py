import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3  # noqa: F401
import numpy as np
from sklearn.datasets import make_s_curve
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector



def swissdatafold(geo, n_components=2,affinity_kwds=1.1, affinity_method="gaussian"):
    X=geo.X
    idx_plot=geo.idx_plot
    X_color=geo.X_color
    radius=affinity_kwds

    X_pcm = pfold.PCManifold(X)

    X_pcm.kernel.epsilon = radius

    X_pcm.optimize_parameters()

    print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")





    if affinity_method == "gaussian":
        dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(
            epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)
        ),
            n_eigenpairs=n_components,
        )




    dmap = dmap.fit(X_pcm)
    evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

    plot_pairwise_eigenvector(
        eigenvectors=dmap.eigenvectors_[idx_plot, :],
        n=1,
        fig_params=dict(figsize=[15, 15]),
        scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot]),
    )
    
    selection = LocalRegressionSelection(
        intrinsic_dim=2, n_subsample=500, strategy="dim"
    ).fit(dmap.eigenvectors_)
    print(f"Found parsimonious eigenvectors (indices): {selection.evec_indices_}")

    target_mapping = selection.transform(dmap.eigenvectors_)

    f, ax = plt.subplots(figsize=(15, 9))
    ax.scatter(
        target_mapping[idx_plot, 0],
        target_mapping[idx_plot, 1],
        c=X_color[idx_plot],
        cmap=plt.cm.Spectral,
    )

