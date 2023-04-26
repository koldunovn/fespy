import xarray as xr
import xarray as xr
from scipy.spatial import cKDTree
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
except ImportError:
    print("Cartopy is not installed, plotting is not available.")


def lon_lat_to_cartesian(lon, lat, R=6371000):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R. Taken from http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z


def create_indexes_and_distances(model_lon, model_lat, lons, lats, k=1, workers=1):
    """
    Creates KDTree object and query it for indexes of points in FESOM mesh that are close to the
    points of the target grid. Also return distances of the original points to target points.
    Parameters
    ----------
    mesh : fesom_mesh object
        pyfesom mesh representation
    lons/lats : array
        2d arrays with target grid values.
    k : int
        k-th nearest neighbors to return.
    n_jobs : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1.
    Returns
    -------
    distances : array of floats
        The distances to the nearest neighbors.
    inds : ndarray of ints
        The locations of the neighbors in data.
    """
    model_lon = xarray_to_numpy(model_lon)
    model_lat = xarray_to_numpy(model_lat)
    lons = xarray_to_numpy(lons)
    lats = xarray_to_numpy(lats)

    xs, ys, zs = lon_lat_to_cartesian(model_lon, model_lat)
    xt, yt, zt = lon_lat_to_cartesian(lons.flatten(), lats.flatten())

    tree = cKDTree(list(zip(xs, ys, zs)))
    distances, inds = tree.query(list(zip(xt, yt, zt)), k=k)

    return distances, inds


def interpolate_kdtree2d(
    data_in,
    lons,
    distances,
    inds,
    radius_of_influence=100000,
    mask_zero=True,
):
    """
    Interpolate data from source target grid using KDTree.
    Parameters
    ----------
    data_in : array
        1d array with data from FESOM mesh
    lons : array
        2d array with target grid longitudes
    distances : array
        2d array with distances from target grid points to FESOM mesh points
    inds : array
        2d array with indexes of FESOM mesh points that are closest to target grid points
    radius_of_influence : float, optional
        Radius of influence for interpolation. Default: 100000.
    mask_zero : bool, optional
        If True, mask all values that are equal to zero. Default: True.
    Returns
    -------
    interpolated : array
        2d array with interpolated data
    """
    data_in = xarray_to_numpy(data_in)
    lons = xarray_to_numpy(lons)

    if len(data_in.shape) > 1:
        raise ValueError("data_in must be 1d array")

    interpolated = data_in[inds]
    interpolated[distances >= radius_of_influence] = np.nan
    if mask_zero:
        interpolated[interpolated == 0] = np.nan
    interpolated.shape = lons.shape

    return interpolated


def interp_kd(
    data_in,
    model_lon,
    model_lat,
    res=1,
    lon_target=None,
    lat_target=None,
    distances=None,
    inds=None,
    k=1,
    workers=1,
    radius_of_influence=100000,
    mask_zero=True,
):
    """
    Interpolate data from source target grid using KDTree.
    Parameters
    ----------
    data_in : array
        1d array with data
    model_lon : array
        1d array with model longitudes
    model_lat : array
        1d array with model latitudes
    res : float, optional
        Resolution of the target grid. Default: 1 degree.
    lon_target/lat_target : array, optional
        1d arrays with target grid longitudes/latitudes. If not provided,
        will be generated using res. Default: None.
    distances/inds : array, optional
        2d arrays with distances/indexes of model grid points that are closest to target grid points.
        If not provided, will be generated. Default: None.
    k : int, optional
        k-th nearest neighbors to return. Default: 1.
    workers : int, optional
        Number of jobs to schedule for parallel processing. Default: 1.
    radius_of_influence : float, optional
        Radius of influence for interpolation. Default: 100000 meters.
    mask_zero : bool, optional
        If True, mask all values that are equal to zero. Default: True.
    Returns
    -------
    interpolated : array
        2d array with interpolated data
    """
    if lon_target is None:
        lon = np.arange(-180, 180, res)
        lon = np.append(lon, np.abs(lon[0]))
    if lat_target is None:
        lat = np.arange(-90, 90, res)
        lat = np.append(lat, np.abs(lat[0]))
        lon_target, lat_target = np.meshgrid(lon, lat)
    if distances is None or inds is None:
        distances, inds = create_indexes_and_distances(
            model_lon, model_lat, lon_target, lat_target, k=k, workers=workers
        )

    interpolated = interpolate_kdtree2d(
        data_in,
        lon_target,
        distances,
        inds,
        radius_of_influence=radius_of_influence,
        mask_zero=mask_zero,
    )

    if isinstance(data_in, xr.DataArray):
        variable_name = data_in.name
        if "time" in data_in.coords:
            time = data_in.time
        depth = get_depths(data_in)
    else:
        variable_name = "data"
        time = None
        depth = None
    interpolated_array = create_data_array(
        interpolated, lon_target, lat_target, variable_name, time=time, depth=depth
    )
    return interpolated_array


def get_depths(data_in):
    possibe_depths = ["depth", "deptht", "depthu", "depthv", "depthw", "nz1", "nz"]
    for depth in possibe_depths:
        if depth in data_in.coords:
            return data_in[depth].values
    return None


def create_data_array(data, lon, lat, variable_name="data", time=None, depth=None):
    """
    Create xarray DataArray from 2d or 3d or 4d array.
    Parameters
    ----------
    data : array
        2d or 3d or 4d array with data
    lon : array
        2d array with longitudes
    lat : array
        2d array with latitudes
    variable_name : str, optional
        Name of the variable. Default: "data".
    time : array, optional
        1d array with time values. Default: None.
    depth : array, optional
        1d array with depth values. Default: None.
    Returns
    -------
    out_data : xarray DataArray
        DataArray with data

    """
    if data.shape != lon.shape or data.shape != lat.shape:
        raise ValueError("data, lon, lat must have the same shape")

    if len(data.shape) == 2:
        data = data[np.newaxis, np.newaxis, :]
    elif len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]
    elif len(data.shape) > 4:
        raise ValueError("data must have 2, 3 or 4 dimensions")

    x = lon[0, :]
    y = lat[:, 0]
    if time is None:
        time = np.arange(data.shape[0])
    if depth is None:
        depth = np.arange(data.shape[1])

    time = np.atleast_1d(time)
    depth = np.atleast_1d(depth)

    out_data = xr.Dataset(
        {variable_name: (["time", "depth", "lat", "lon"], data)},
        coords={
            "time": time,
            "depth": depth,
            "lon": (["lon"], x),
            "lat": (["lat"], y),
            "longitude": (["lat", "lon"], lon),
            "latitude": (["lat", "lon"], lat),
        },
        # attrs=data.attrs,
    )
    return out_data


def plot(
    data,
    lon,
    lat,
    res=1,
    projection="PlateCarree",
    domain=None,
    add_land=True,
    set_global=False,
    **kwargs
):

    working_projection = getattr(ccrs, projection)()

    if projection == "NorthPolarStereo":
        domain = domain or [-180, 180, 60, 90]
    elif projection == "SouthPolarStereo":
        domain = domain or [-180, 180, -90, -60]
    domain = domain or [-180, 180, -90, 90]
    a = interp_kd(data, lon, lat, res=res)
    variable = list(a.data_vars.keys())[0]
    p = a[variable].plot(
        subplot_kws=dict(projection=working_projection),
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"shrink": 0.9, "pad": 0.01, "orientation": "horizontal"},
        **kwargs
    )
    if add_land:
        p.axes.add_feature(
            cfeature.GSHHSFeature(levels=[1], scale="low", facecolor="lightgray")
        )
    if set_global:
        p.axes.set_global()
    else:
        p.axes.set_extent(domain, crs=ccrs.PlateCarree())

    p.axes.coastlines()
    return p


def plot_region():
    """
    Version of the plot, that is more efficient for plotting a region, as the interpolation only done for the region.
    """


def xarray_to_numpy(data):
    """Convert xarray DataArray to numpy array"""
    if isinstance(data, xr.DataArray):
        return data.values
    else:
        return data
