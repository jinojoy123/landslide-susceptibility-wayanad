import rasterio
import numpy as np
import joblib
from rasterio.windows import Window
from rasterio.enums import Resampling

model = joblib.load('logreg_model.joblib')
scaler = joblib.load('logreg_scaler.joblib')

rasters = {
    'curv': 'Curvature.tif',
    'relief': 'Relief.tif',
    'slope': 'Slope.tif',
    'twi': 'TWI.tif'
}
# open all rasters
srcs = [rasterio.open(p) for p in rasters.values()]
# basic checks (shape/affine)
ref = srcs[0]
meta = ref.meta.copy()
meta.update(dtype='float32', count=1, compress='lzw')

out_path = 'logreg_probability_map.tif'
with rasterio.open(out_path, 'w', **meta) as dst:
    # iterate windows (block windows of ref)
    for ji, window in ref.block_windows(1):
        # read each predictor for this window
        band_stack = []
        mask = None
        for s in srcs:
            arr = s.read(1, window=window)
            band_stack.append(arr)
            if mask is None:
                mask = np.isnan(arr)  # start mask
            else:
                mask = mask | np.isnan(arr)
        # shape = (n_bands, h, w)
        stack = np.stack(band_stack, axis=0)
        n_bands, h, w = stack.shape
        # reshape to (n_pixels, n_bands)
        pixels = stack.reshape(n_bands, -1).T  # shape (N,4)
        # find valid pixels (no nodata)
        valid = ~np.any(np.isnan(pixels), axis=1)
        probs = np.full(pixels.shape[0], np.nan, dtype=np.float32)
        if valid.any():
            Xpix = pixels[valid]
            # scale
            Xpixs = scaler.transform(Xpix)
            # predict probabilities
            probs_valid = model.predict_proba(Xpixs)[:,1]
            probs[valid] = probs_valid
        # reshape back to (h,w)
        prob_block = probs.reshape(h, w).astype(np.float32)
        dst.write(prob_block, 1, window=window)

# close sources
for s in srcs:
    s.close()
