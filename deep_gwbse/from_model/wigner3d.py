import numpy as np
import logging
import time
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import plotly.graph_objects as go

from deep_gwbse.from_model.model_util import time_watch


class WignerXYZ:
    """
    True 3D Wigner-Seitz interpolation for real-space wavefunction / density.
    """

    def __init__(
        self,
        lattice: np.ndarray,
        FFT_grid_shape: np.ndarray,
        AngstromPerPixel: float = 0.2,
        **kwargs,
    ):
        """
        Args:
            lattice (np.ndarray): lattice vectors in Cartesian (Å), shape (3,3)
            FFT_grid_shape (np.ndarray): original FFT grid (Nx, Ny, Nz)
            AngstromPerPixel (float): uniform output grid spacing in Å
        kwargs:
            upsampling_factor (float): optional FFT grid upsampling
            lattice_replica (int): range of lattice replicas (default 2)
        """

        self.lattice = lattice
        self.upsampling_factor = kwargs.get("upsampling_factor", 1)
        self.AngstromPerPixel = AngstromPerPixel
        replica = kwargs.get("lattice_replica", 2)

        # --------------------------------------------------
        # Build lattice replica points (3D)
        # --------------------------------------------------
        self.lattice_points = np.array([
            i * lattice[0] + j * lattice[1] + k * lattice[2]
            for i in range(-replica, replica + 1)
            for j in range(-replica, replica + 1)
            for k in range(-replica, replica + 1)
        ])

        # --------------------------------------------------
        # Fractional grid → Cartesian grid
        # --------------------------------------------------
        Nx, Ny, Nz = FFT_grid_shape
        Nx_u = round(Nx * self.upsampling_factor)
        Ny_u = round(Ny * self.upsampling_factor)
        Nz_u = round(Nz * self.upsampling_factor)

        fx = np.linspace(0, 1, Nx_u, endpoint=False)
        fy = np.linspace(0, 1, Ny_u, endpoint=False)
        fz = np.linspace(0, 1, Nz_u, endpoint=False)

        frac_grid = np.array(
            [[x, y, z] for x in fx for y in fy for z in fz]
        )

        self.grid_points = frac_grid @ lattice  # (N,3)
        self.grid_points_folded = self.fold_to_wigner_seitz(
            self.grid_points, self.lattice_points
        )

        # --------------------------------------------------
        # Uniform Cartesian Wigner grid
        # --------------------------------------------------
        mins = self.grid_points_folded.min(axis=0)
        maxs = self.grid_points_folded.max(axis=0)

        self.xi = np.arange(mins[0], maxs[0], AngstromPerPixel)
        self.yi = np.arange(mins[1], maxs[1], AngstromPerPixel)
        self.zi = np.arange(mins[2], maxs[2], AngstromPerPixel)

        self.XI, self.YI, self.ZI = np.meshgrid(
            self.xi, self.yi, self.zi, indexing="ij"
        )

        logging.debug(
            f"WignerXYZ initialized. Output grid: {self.XI.shape}"
        )

    @staticmethod
    def fold_to_wigner_seitz(points, lattice_points):
        """
        Fold Cartesian points to the nearest lattice replica
        """
        tree = cKDTree(lattice_points)
        _, idx = tree.query(points)
        return points - lattice_points[idx]

    # ======================================================
    # Interpolation
    # ======================================================
    @time_watch
    def WignerInterpolate(self, wf_3D: np.ndarray, method="linear"):
        """
        Args:
            wf_3D (np.ndarray): real-space density/wavefunction (Nx,Ny,Nz)
            method (str): 'linear' or 'nearest'
        Returns:
            zi (np.ndarray): interpolated 3D Wigner density
        """
        assert np.isrealobj(wf_3D), "wf_3D must be real"

        charge_total = wf_3D.sum()

        # Optional upsampling
        density = zoom(
            wf_3D,
            (self.upsampling_factor,) * 3,
            order=3 if method == "linear" else 1,
        )

        values = density.flatten()
        query_points = np.column_stack([
            self.XI.flatten(),
            self.YI.flatten(),
            self.ZI.flatten(),
        ])

        zi = griddata(
            self.grid_points_folded,
            values,
            query_points,
            method=method,
        )

        zi = zi.reshape(self.XI.shape)

        # Charge conservation
        charge_new = np.nansum(zi)
        zi *= charge_total / charge_new

        self.zi = zi
        return zi

    # ======================================================
    # Fast nearest-neighbor version
    # ======================================================
    @time_watch
    def Wigner_fast_nearest(self, wf_3D, max_distance=0.2):
        """
        Fast nearest-neighbor interpolation using KDTree
        """
        assert np.isrealobj(wf_3D)

        charge_total = wf_3D.sum()

        density = zoom(
            wf_3D,
            (self.upsampling_factor,) * 3,
            order=1,
        )

        values = density.flatten()
        tree = cKDTree(self.grid_points_folded)

        query = np.column_stack([
            self.XI.flatten(),
            self.YI.flatten(),
            self.ZI.flatten(),
        ])

        dist, idx = tree.query(query, k=1)
        zi = values[idx]
        zi[dist > max_distance] = np.nan
        zi = zi.reshape(self.XI.shape)

        charge_new = np.nansum(zi)
        zi *= charge_total / charge_new

        self.zi = zi
        return zi

def plot_wigner_3d(XI, YI, ZI, density):
    """
    Plot a 3D volume rendering of Wigner density.

    Args:
        XI, YI, ZI (np.ndarray): meshgrid arrays (Nx, Ny, Nz)
        density (np.ndarray): scalar field (Nx, Ny, Nz)
    """

    density = np.where(np.isnan(density), 0, density)
    density = np.where(density < 0.16*density.max(), 0, density)

    fig = go.Figure(
        data=go.Volume(
            x=XI.flatten(),
            y=YI.flatten(),
            z=ZI.flatten(),
            value=density.flatten(),
            colorscale="Viridis",
            opacity=0.15,
            surface_count=20,
        )
    )

    fig.update_layout(
        title="3D Wigner-Seitz Density",
        scene=dict(
            xaxis_title="x (Å)",
            yaxis_title="y (Å)",
            zaxis_title="z (Å)",
            aspectmode="data",
        ),
    )

    fig.show()


if __name__ == "__main__":
    import interface
    import matplotlib.pyplot as plt
    au2ang = 0.52917721067

    # wf = interface.wfn('/pscratch/sd/b/bwhou/16-NCS/DeepGWBSE/reply/01-bulk-calculation/small_test/flows/001-mp-1019089_KMgAs/02-wfn/wfn.h5')
    # wf = interface.wfn('../../examples/flows/mat-5/02-wfn/wfn.h5')
    wf = interface.wfn('/pscratch/sd/b/bwhou/16-NCS/DeepGWBSE/reply/01-bulk-calculation/flows/002-mp-1185959_MgS2/02-wfn/wfn.h5')

    w00_3D = abs(wf.get_dataset(cell_slab_truncation=None, AngstromPerPixel=0.2)['wfn'][0,4,:,:,:])
    lattice = wf.crystal["avec"] * wf.crystal["alat"] * au2ang
    FFT_grid_shape = w00_3D.shape

    wigner = WignerXYZ(
        lattice,
        FFT_grid_shape,
        AngstromPerPixel=0.2,
        upsampling_factor=4,
    )

    zi = wigner.Wigner_fast_nearest(w00_3D)

    plot_wigner_3d(wigner.XI, wigner.YI, wigner.ZI, zi)