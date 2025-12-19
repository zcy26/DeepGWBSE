import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay, cKDTree
from scipy.interpolate import griddata, RBFInterpolator

import torch.nn.functional as F
from scipy.ndimage import zoom
import time
from scipy.interpolate import LinearNDInterpolator
import logging
import plotly.graph_objects as go
from from_model.model_util import time_watch
au2ang = 0.52917721067
class WignerXY:
    def __init__(self, lattice: np.ndarray, FFT_grid_shape: np.array, 
                 AngstromPerPixel:float=0.1,**kwargs):
        """
        Args:
            lattice (np.ndarray): lattice vectors in Cartesian (A). (3, 3)
            FFT_grid_shape (np.array): FFT grid in fractional coordinates. (Rx, Ry, Rz)
            AngstromPerPixel (float): 
                Wigner2D output a uniform grid matrix (nxn), we define AngstromPerPixel as
                the distance between two adjacent points in the grid. This is used to make 
                sure CNN kernel can always be applied to the same area when scanning the grid.
                e.g. if AngstromPerPixel = 0.1, and the grid is 40x40, then the output will be 4Ax4A area.
        kwargs:
            upsampling_factor (float): upsampling wavefunction in original grid (not recommended).
        """
        self.upsampling_factor = kwargs.get('upsampling_factor', 1)
        self.AreaPerPixel = AngstromPerPixel

        # Lattice
        self.lattice = lattice
        self.lattice_2D = lattice[:2, :2] # 2D (2, 2)
        grid_size = 2
        self.lattice_points = np.array([
            m * self.lattice_2D[0] + n * self.lattice_2D[1]
            for m in range(-grid_size, grid_size+1)
            for n in range(-grid_size, grid_size+1)])

        # Real Space Grid
        self.upsampling_FFT_grid_size = (round(FFT_grid_shape[0] * self.upsampling_factor),
                                         round(FFT_grid_shape[1] * self.upsampling_factor))
        grid_x = np.linspace(0, 1, self.upsampling_FFT_grid_size[0], endpoint=False)
        grid_y = np.linspace(0, 1, self.upsampling_FFT_grid_size[1], endpoint=False)
        grid_points_frac = np.array([[x, y] for x in grid_x for y in grid_y])
        # real-space primitive cell grid in Cartesian (A)
        self.grid_points = np.dot(grid_points_frac, self.lattice_2D) 
        # real-space wigner cell grid in Cartesian (A) 
        self.grid_points_folded = np.array([self.fold_to_wigner_seitz(pt, self.lattice_points) for pt in self.grid_points])

        self.x_min, self.x_max = self.grid_points_folded[:, 0].min(), self.grid_points_folded[:, 0].max()
        self.y_min, self.y_max = self.grid_points_folded[:, 1].min(), self.grid_points_folded[:, 1].max()
        xi = np.arange(self.x_min, self.x_max, self.AreaPerPixel)
        yi = np.arange(self.y_min, self.y_max, self.AreaPerPixel)
        self.xi, self.yi = np.meshgrid(xi, yi)

        logging.debug(f'Wigner2D initialized. Output grid shape: {self.xi.shape}, AreaPerPixel: {self.AreaPerPixel} A/pixel')

    @classmethod
    def fold_to_wigner_seitz(cls, points, lattice_points):
        tree = cKDTree(lattice_points)
        _, indices = tree.query(points)
        return points - lattice_points[indices]

    @time_watch
    def WignerInterpolate(self, wf_3D: np.ndarray):
        assert np.isrealobj(wf_3D), "wf_3D must be a real matrix"
        """
        Args:
            wf_3D (np.ndarray): 
                wavefunction in 3D fractional coordinate of a KS state. (FFT_grid_shape)
            **kwargs:
                method (str): interpolation method. Default is 'linear'.
        Returns:
            Only conduct Wigner interpolation on 2D slice of wf_3D (The z axis is preserved, like batch)
            wf_wigner (np.ndarray): Wigner interpolation result. (xi.shape, z)
        """
        self.wf_3D = wf_3D
        charge_total = np.sum(self.wf_3D)
        self.density_matrix = zoom(self.wf_3D, (self.upsampling_factor, self.upsampling_factor, 1), order=3)
        zi = []
        for z_idex in range(self.density_matrix.shape[2]):
            slice_data = self.density_matrix[:, :, z_idex].flatten()
            zi_slice = griddata(self.grid_points_folded, slice_data, (self.xi, self.yi), method='linear')
            zi.append(zi_slice)
        self.zi = np.array(zi)
        self.zi = np.transpose(self.zi, (1, 2, 0))

        charge_total_new = np.where(np.isnan(self.zi), 0, self.zi).sum()
        self.zi = np.where(np.isnan(self.zi), np.nan, self.zi * charge_total / charge_total_new)
        return self.zi
    
    @time_watch        
    def Wigner_fast_nearest_interpolation(self, wf_3D, max_distance=0.1):

        """
        Args:
            wf_3D (np.ndarray): Input wavefunction data.
            max_distance (float): Maximum allowed distance for interpolation. 
                                Points farther than this will be set to NaN.
        Returns:
            np.ndarray: Interpolated result with NaNs where no nearby data exists.
        """
        assert np.isrealobj(wf_3D), "wf_3D must be a real matrix"
        self.wf_3D = wf_3D
        charge_total = np.sum(self.wf_3D)

        start_time = time.time()

        # TODO: ensure periodic boundary condition when interpolating
        # pad_size = 4 # 2 is enough for periodic boundary condition
        # circular_padded_density_matrix = np.pad(self.wf_3D, ((pad_size,pad_size), (pad_size,pad_size),(0,0)), mode='wrap')
        # density_matrix_upsampled = zoom(circular_padded_density_matrix, (self.upsampling_factor, self.upsampling_factor, 1), order=1)
        # resized_pad = round(pad_size * self.upsampling_factor)
        # self.density_matrix = density_matrix_upsampled[resized_pad:-resized_pad, resized_pad:-resized_pad, :]

        self.density_matrix = zoom(self.wf_3D, (self.upsampling_factor, self.upsampling_factor, 1), order=1)
        logging.debug(f'Zooming completed in {time.time() - start_time:.4f} seconds')

        start_time = time.time()
        zi = []
        tree = cKDTree(self.grid_points_folded)
        query_points = np.vstack([self.xi.flatten(), self.yi.flatten()]).T
        distances, indices = tree.query(query_points, k=1)
        logging.debug(f'KDTree query completed in {time.time() - start_time:.4f} seconds')

        start_time = time.time()
        for z_index in range(self.density_matrix.shape[2]):
            interpolated_values = self.density_matrix[:, :, z_index].flatten()[indices]
            
            # Set points too far from known data to NaN
            interpolated_values[distances > max_distance] = np.nan  

            zi_slice = interpolated_values.reshape(self.xi.shape)
            zi.append(zi_slice)
        logging.debug(f'Interpolation completed in {time.time() - start_time:.4f} seconds')

        start_time = time.time()
        self.zi = np.array(zi).transpose(1, 2, 0)
        charge_total_new = np.where(np.isnan(self.zi), 0, self.zi).sum()
        self.zi = np.where(np.isnan(self.zi), np.nan, self.zi * charge_total / charge_total_new)
        logging.debug(f'Normalization completed in {time.time() - start_time:.4f} seconds')

        return self.zi


    def plot(self, go_3D:bool=False, **kwargs):
        assert hasattr(self, 'zi'), "Please run WignerInterpolate first"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im1 = axes[0].imshow(self.zi.sum(axis=2), extent=[self.x_min, self.x_max, self.y_min, self.y_max], 
                            origin='lower', cmap='viridis', aspect='auto')
        axes[0].set_title("Wigner Cell")
        plt.colorbar(im1, ax=axes[0], label="Interpolated Density")
        im2 = axes[1].imshow(self.wf_3D.sum(axis=2), extent=[self.x_min, self.x_max, self.y_min, self.y_max], 
                            origin='lower', cmap='viridis', aspect='auto')
        axes[1].set_title("Primitive Cell")
        plt.colorbar(im2, ax=axes[1], label="w00_3D Density")
        plt.suptitle("Comparison of Upsampled Density Map and w00_3D in Wigner-Seitz Cell")
        plt.show()

        ###########
        if go_3D:
            xi, yi, zi = np.mgrid[
            self.grid_points_folded[:, 0].min():self.grid_points_folded[:, 0].max():self.zi.shape[0]*1j,
            self.grid_points_folded[:, 1].min():self.grid_points_folded[:, 1].max():self.zi.shape[1]*1j,
            0:self.lattice[2,2]:self.zi.shape[2]*1j]

            fig = go.Figure(data=go.Volume(
                x=xi.flatten(), y=yi.flatten(), z=zi.flatten(),
                value=self.zi.flatten(),
                opacity=0.2, surface_count=20, colorscale='Viridis'
            ))
            fig.update_layout(title='3D Wigner-Seitz Density Visualization')
            fig.show()

if __name__ == '__main__':
    import interface
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    # logging.basicConfig(level=logging.INFO, format='%(message)s')

    wf = interface.wfn('../../examples/flows/mat-5/02-wfn/wfn.h5')
    w00_3D = abs(wf.get_dataset(cell_slab_truncation=15, AngstromPerPixel_z=0.2)['wfn'][0,3,:,:,:])
    lattice = wf.crystal['avec'] * wf.crystal['alat'] * au2ang
    FFT_grid_shape = w00_3D.shape
    shift = 14
    w00_3D = np.concatenate([w00_3D[shift:],w00_3D[:shift]], axis=0)

    wigner = WignerXY(lattice, FFT_grid_shape, AngstromPerPixel=0.05 ,upsampling_factor=1)
    wigner.WignerInterpolate(w00_3D)
    wigner.plot(go_3D=False)

    wigner = WignerXY(lattice, FFT_grid_shape, AngstromPerPixel=0.05 ,upsampling_factor=8)
    wigner.Wigner_fast_nearest_interpolation(w00_3D, max_distance=0.05)
    wigner.plot(go_3D=True)


    assert abs(np.where(np.isnan(wigner.zi),0,wigner.zi).sum() - 0.05038487949550353) < 1e-6, "Wigner Unit Test Failed"

