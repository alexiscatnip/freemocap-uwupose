from freemocap import fmc_anipose
import numpy as np
from rich.console import Console
console=Console()

# from numba import jit

# @jit(nopython=True)
def reconstruct3D(session, data_nCams_nImgPts_XYC, confidenceThreshold=0.8):
    """
    Take a specifically formatted data array, and based on the cameras'
    calibration.toml, reconstruct 3D key-points
    """

    if (
        session.cgroup is None
    ):  # looks like we failed to load the calibration.toml in an earlier stage
        return None, None

    nCams, nImgPts, nDims = data_nCams_nImgPts_XYC.shape

    if nCams == 1:
        print("called reconstruct3d on dataArray that has data from only 1 camera.")
        return None, None

    if nDims == 3:
        dataOG = data_nCams_nImgPts_XYC.copy()

        for camNum in range(nCams):
                          
            thisCamX = data_nCams_nImgPts_XYC[camNum, :,0 ]
            thisCamY = data_nCams_nImgPts_XYC[camNum, :,1 ]
            thisCamConf = data_nCams_nImgPts_XYC[camNum, :, 2]

            thisCamX[thisCamConf < confidenceThreshold] = np.nan
            thisCamY[thisCamConf < confidenceThreshold] = np.nan

    if nDims == 2:
        data_nCams_nImgPts_XY = data_nCams_nImgPts_XYC
    elif nDims == 3:
        data_nCams_nImgPts_XY = np.squeeze(data_nCams_nImgPts_XYC[:, :, 0:2]) #remove confidence

    dataFlat_nCams_nTotalPoints_XY = data_nCams_nImgPts_XY # data_nCams_nImgPts_XY.reshape(nCams, -1, 2)  # reshape data to collapse across 'frames' so it becomes [numCams, numFrames*numPoints, XY]

    # console.print('Reconstructing 3d points...')
    data3d_flat = session.cgroup.triangulate(dataFlat_nCams_nTotalPoints_XY, progress=False)

    dataReprojerr_flat = session.cgroup.reprojection_error(data3d_flat, dataFlat_nCams_nTotalPoints_XY, mean=True)
    dataReprojerr_C_N_2 = \
        session.cgroup.reprojection_error(data3d_flat,
                                          dataFlat_nCams_nTotalPoints_XY,
                                          mean=False)

    data_fr_mar_xyz = data3d_flat.reshape(nImgPts, 3)
    dataReprojErr = dataReprojerr_flat.reshape(nImgPts)

    return data_fr_mar_xyz, dataReprojErr, dataReprojerr_C_N_2
