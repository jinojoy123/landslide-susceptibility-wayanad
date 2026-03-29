import rasterio
import numpy as np
import joblib
from rasterio.windows import Window
from rasterio.enums import Resampling

# -------------------------------
# Load model (NO scaler for RF)
# -------------------------------
rf_model = joblib.load('rf_model.joblib')

rasters = {
    'curv': 'Curvature.tif',
    'relief': 'Relief.tif',
    'slope': 'Slope.tif',
    'twi': 'TWI.tif'
}

# -------------------------------
# Open all rasters
# -------------------------------
srcs = [rasterio.open(p) for p in rasters.values()]
ref = srcs[0]

# Copy metadata from reference raster
meta = ref.meta.copy()
meta.update(dtype='float32', count=1, compress='lzw')

# -------------------------------
# Output raster path
# -------------------------------
out_path = "rf_probability_map.tif"

with rasterio.open(out_path, "w", **meta) as dst:

    # Loop through windows (blocks)
    for ji, window in ref.block_windows(1):

        # Read block from each raster
        band_stack = []
        mask = None

        for s in srcs:
            arr = s.read(1, window=window)
            band_stack.append(arr)

            if mask is None:
                mask = np.isnan(arr)
            else:
                mask = mask | np.isnan(arr)

        # Convert stack to (N, 4)
        stack = np.stack(band_stack, axis=0)      # (4, h, w)
        n_bands, h, w = stack.shape
        pixels = stack.reshape(n_bands, -1).T     # (N, 4)

        # Identify valid pixels (no NaN)
        valid = ~np.any(np.isnan(pixels), axis=1)

        # Prepare output array
        probs = np.full(pixels.shape[0], np.nan, dtype=np.float32)

        if valid.any():
            X_valid = pixels[valid]

            # Predict using Random Forest
            probs_valid = rf_model.predict_proba(X_valid)[:, 1]

            probs[valid] = probs_valid.astype(np.float32)

        # Reshape back to (h, w)
        prob_block = probs.reshape(h, w)

        # Write to output raster
        dst.write(prob_block.astype("float32"), 1, window=window)

# Close input rasters
for s in srcs:
    s.close()

print("Saved:", out_path)
