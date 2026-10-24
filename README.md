# nisar_pytools

Open source Python tools for working with NISAR datasets.

## About

`nisar_pytools` opens NASA [NISAR](https://nisar.jpl.nasa.gov/) (NASA-ISRO
Synthetic Aperture Radar) HDF5 products as lazy `xarray` objects. Search for a
product, open it, and pull out a layer without learning the HDF5 layout.

Reading products needs nothing beyond the base install. Building them is
covered in the [processing reference](docs/processing.md).

### Supported Products

- **GSLC** - Geocoded Single Look Complex
- **GUNW** - Geocoded Unwrapped Interferogram

`find_nisar` searches for other types (RSLC, GCOV, RIFG, RUNW, ROFF, GOFF),
and RSLC products can be streamed and cropped. Full reader support for
more types will come over time.

## Quick start

```python
from nisar_pytools import open_nisar
from nisar_pytools.utils.metadata import get_gunw

dt = open_nisar("NISAR_L2_PR_GUNW_...h5")

# A whole layer, masked. Every variable in it shares one grid.
ds = get_gunw(dt)
ds.unwrappedPhase.plot()
ds.coherenceMagnitude.plot()
```

```
<xarray.Dataset>
Dimensions:  (y: 4347, x: 4410)
Data variables:
    unwrappedPhase                    (y, x) float32
    coherenceMagnitude                (y, x) float32
    connectedComponents               (y, x) float32
    ionospherePhaseScreen             (y, x) float32
    ionospherePhaseScreenUncertainty  (y, x) float32
```

Nothing is read from disk until you plot or `.compute()`, so this works the
same on a 2 GB product as on a small one.

## Installation

Needs Python 3.10+ and a NASA Earthdata login to download from ASF.
[Miniforge](https://github.com/conda-forge/miniforge) is recommended.

```sh
pip install nisar-pytools
```

With optional extras:

```sh
pip install nisar-pytools[dem]       # DEM fetching (dem_stitcher)
pip install nisar-pytools[dolphin]   # dolphin InSAR time-series prep
pip install nisar-pytools[viz]       # Visualization (matplotlib)
pip install nisar-pytools[all]       # Everything available from PyPI
```

**From source (for development):**

```sh
git clone https://github.com/zmhoppinen/nisar_pytools.git
cd nisar_pytools
mamba env create -f environment.yml
conda activate nisar_pytools
```

The bundled `environment.yml` also installs the ISCE3 RSLC to GUNW stack.
You only need that for the [processing reference](docs/processing.md).
Reading products does not require it.

## Reading products

### How a NISAR file is laid out

`open_nisar` returns an `xarray.DataTree` that mirrors the HDF5 groups, so a
path in the tree is the same path you would use in `h5dump` or the NISAR
product spec. NISAR stores nothing at the root, so printing the tree shows
`Data variables: (0)` and a single group. That is expected. The data is one
level down:

```
/science/LSAR/identification            <- track, frame, times, polygon (attrs)
/science/LSAR/GUNW/grids/frequencyA/
        unwrappedInterferogram/HH       <- unwrapped phase, coherence, ...
        wrappedInterferogram/HH         <- wrapped interferogram, coherence
        pixelOffsets/HH                 <- along-track / slant-range offsets
```

To see every group that actually holds arrays:

```python
[node.path for node in dt.subtree if node.dataset.data_vars]
```

The accessors below save you from walking this by hand, but the tree is
always there if you want it:

```python
freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
```

### GUNW layers

`get_gunw` returns a whole layer by default, since the variables in one layer
share a grid:

```python
from nisar_pytools.utils.metadata import get_gunw

dt = open_nisar("NISAR_L2_PR_GUNW_...h5")

# Defaults: unwrappedInterferogram / HH, masked.
ds = get_gunw(dt)
ds.unwrappedPhase
ds.coherenceMagnitude
ds.connectedComponents

# A single variable comes back as a DataArray.
unw = get_gunw(dt, variable="unwrappedPhase")

# Raw samples, nothing blanked.
ds_raw = get_gunw(dt, valid_mask=False)
```

The three layers are **not** on a common grid. In a typical product the
unwrapped interferogram and pixel offsets are posted at 80 m, and the wrapped
interferogram at 20 m. Merging them into one Dataset would mean resampling, so
ask for one layer at a time:

```python
wrapped = get_gunw(dt, layer="wrappedInterferogram")   # 20 m grid
offsets = get_gunw(dt, layer="pixelOffsets")           # 80 m grid
```

### GSLC channels

```python
from nisar_pytools.utils.metadata import get_slc

dt = open_nisar("NISAR_L2_PR_GSLC_...h5")

hh = get_slc(dt, polarization="HH")                        # lazy, CRS set, masked
hv = get_slc(dt, polarization="HV")
hh_b = get_slc(dt, polarization="HH", frequency="frequencyB")
hh_raw = get_slc(dt, polarization="HH", valid_mask=False)   # opt out of masking
```

### Metadata

```python
from nisar_pytools.utils.metadata import (
    get_orbit_info, get_bounding_polygon,
)

print(get_orbit_info(dt))         # {'track_number': 77, 'frame_number': 24, ...}
print(get_bounding_polygon(dt))   # shapely Polygon in WGS84
print(hh.rio.crs)                 # EPSG:32611
```

`get_acquisition_time` returns both passes, so it reads the same way whether
the product came from one acquisition or two:

```python
from nisar_pytools.utils.metadata import get_acquisition_time

t = get_acquisition_time(gslc)
t.reference                       # 2025-11-03 12:46:15
t.secondary                       # None, a GSLC is one acquisition

t = get_acquisition_time(gunw)
t.reference                       # 2026-02-15 12:11:20
t.secondary                       # 2026-02-27 12:11:20
(t.secondary - t.reference).days  # 12, the temporal baseline
```

### Valid mask semantics

When `valid_mask=True` (the default), invalid pixels become `NaN`.
Integer-typed variables (e.g. `connectedComponents`, uint16) are promoted
to float so `NaN` fits.

- **GSLC** `mask` (uint8) encodes the subswath number for valid samples.
  - `0` = at least one RSLC pixel in the interpolation window was
    partially-focused or invalid, so the sample is dropped.
  - `1..N` = valid subswath number, kept.
  - `255` = outside the radar acquisition extent, dropped.
- **GUNW** `mask` (uint8) is a three-digit `WRS` integer combining a water
  flag and the reference/secondary RSLC subswath numbers.
  - `W` = reference water flag (1 = water, 0 = land). Water pixels are
    **kept**, so mask water separately if you need to drop it.
  - `R` = reference RSLC subswath number. `0` means the sample is invalid
    in the reference, so it is dropped.
  - `S` = secondary RSLC subswath number. `0` means the sample is invalid
    in the secondary, so it is dropped.
  - `255` = fill outside the acquisition extent, dropped.

  In code: kept where `(mask // 10) % 10 != 0 and mask % 10 != 0 and mask != 255`.

  Each GUNW layer carries its own mask on its own grid. Within a layer, the
  mask is shared across polarizations.

## Command line

### GeoTIFF export

Exports common bands from a NISAR HDF5 to GeoTIFF, ready to drop into QGIS.
The package installs a `nisar_pytools` console script. Run
`nisar_pytools to-geotiff --help` for the full band catalog.

```bash
# Default-all: write every default band for the product, next to the .h5
# GUNW -> unwrapped_phase, wrapped_phase, coherence, ionosphere
# GSLC -> amplitude (10*log10(|SLC|^2) in dB)
nisar_pytools to-geotiff NISAR_L2_PR_GUNW_...h5

# One band, explicit polarization, custom output directory
nisar_pytools to-geotiff NISAR_L2_PR_GUNW_...h5 \
    --band unwrapped_phase --pol HH --output-dir /tmp/gunw_tifs

# Subset a multi-GB GSLC to a lat/lon AOI -- streamed, low memory
nisar_pytools to-geotiff NISAR_L2_PR_GSLC_...h5 \
    --bbox-wgs84 -118.5 41.0 -118.3 41.2

# Same crop in the file's native CRS (UTM meters here)
nisar_pytools to-geotiff NISAR_L2_PR_GSLC_...h5 \
    --bbox 380000 4540000 400000 4565000

# GSLC amplitude on frequency B
nisar_pytools to-geotiff NISAR_L2_PR_GSLC_...h5 --freq B
```

Outputs are tiled GeoTIFFs named `<h5_stem>_<band>_<frequency>_<pol>.tif`.
Writes stream chunk-by-chunk via dask + rioxarray, so a full-resolution
41 GB GSLC processes with ~330 MB peak memory.

### File summary

Prints what a file holds without opening it in Python: product type and
version, file size, acquisition times, track, frame, direction, polarizations,
per-grid shape and resolution, and the extent in native and WGS84 coordinates.

For a GUNW it adds coherence and unwrapped-phase statistics, a
connected-component summary, and the perpendicular and parallel baselines from
the radarGrid cube.

```bash
nisar_pytools info NISAR_L2_PR_GUNW_...h5
nisar_pytools info NISAR_L2_PR_GSLC_...h5 --json
```

## Search and download

```python
from nisar_pytools import find_nisar, download_urls

# aoi          - [xmin, ymin, xmax, ymax], shapely geometry, or
#                {"west": -115, "south": 43, "east": -114, "north": 44}
# product_type - "GSLC", "GUNW", "RSLC", "GCOV", "RIFG", "RUNW", "ROFF", "GOFF"
# path_number  - relative orbit / track number (optional)
# frame        - frame number (optional)
# direction    - "ASCENDING" or "DESCENDING" (optional)
# maturity     - "validated", "provisional" or "beta" (optional)
# include_qa   - if True, include QA files (default False)

all_gslcs = find_nisar(
    aoi=[-115, 43, -114, 44],
    start_date="2025-06-01",
    end_date="2025-12-01",
    product_type="GSLC",
)

# Narrow to one track and direction for time-series analysis
track_77 = find_nisar(
    aoi=[-115, 43, -114, 44],
    start_date="2025-06-01",
    end_date="2025-12-01",
    product_type="GSLC",
    path_number=77,
    direction="ASCENDING",
)

# Download in parallel with automatic resumption and HDF5 validation
fps = download_urls(track_77, "local/gslcs/")
```

Interrupted downloads remain as `<filename>.part` with a `.part.json` sidecar.
Run the same call again to resume, including after restarting Python. The server
must support HTTP Range and provide a strong ETag or Last-Modified validator;
otherwise the file restarts safely. A server that ignores Range also triggers a
fresh download. Do not run two download calls into the same files simultaneously.

The final filename appears only after checking the advertised total length and
NISAR HDF5 structure. A `.download.json` sidecar records the completed length and
filesystem identity for later checks. This also prevents an interrupted final
rename from validating an older file with the new file's metadata. Existing HDF5
files are validated before being skipped. A transfer
with no trustworthy total length is kept as a partial file and is not reported as
successful. `validate=False` disables HDF5 validation, but transfer length checks
still apply. `reprocess=True` starts a fresh transfer. Failed downloads are omitted
from the returned list, so check its length when a complete set is required.

```python
import logging

logging.basicConfig(level=logging.INFO)  # Progress every 30 seconds
fps = download_urls(track_77, "local/gslcs/", max_workers=2, retries=5, timeout=120)
```

Sidecars and downloader messages do not store download URLs or signed query
strings. Authentication continues to use the standard Requests environment,
including your NASA Earthdata credentials in `.netrc`.

NISAR products are published at several processing maturities, which cover
different date ranges rather than reprocessing the same acquisitions. Leaving
`maturity` unset searches all of them and warns when results span more than
one.

### Download a GSLC subset

Use `download_gslc_subset` to fetch one polarization and its validity layers
inside a WGS84 bounding box. HTTP byte-range requests read the required HDF5
chunks; the full source granule does not need to be downloaded. The resulting
HDF5 is explicitly marked as a **derived subset**, with source provenance and
cropped coordinates. It is not a complete official NISAR product.
This path supports chunked LSAR GSLC layers on regular north-up grids and
requires a server that provides HTTP Range responses and a strong ETag.
Only selected identification, grid and direct processing/input/orbit metadata
are retained; calibration and other ancillary metadata are omitted, with this
scope recorded inside the file.

```python
from nisar_pytools import download_gslc_subset

report = download_gslc_subset(
    gslc_url,
    "local/gslcs/scene_derived_subset.h5",
    bbox=(92.9, 31.6, 94.7, 33.2),  # WEST, SOUTH, EAST, NORTH
    cache_dir="local/gslcs/range_cache",
    reuse_prefix_dir="local/gslcs",  # Optional existing whole-granule .part files
    frequency="A",
    polarization="HH",
    max_workers=2,
)
```

The same operation is available from the command line:

```bash
nisar_pytools download-gslc-subset "$GSLC_URL" \
    --output local/gslcs/scene_derived_subset.h5 \
    --bbox-wgs84 92.9 31.6 94.7 33.2 \
    --cache-dir local/gslcs/range_cache \
    --reuse-prefix-dir local/gslcs --freq A --pol HH
```

Repeat the command after an interruption to reuse verified cached ranges.
Whole-granule partial downloads are read only when their recorded source and
validator match the remote resource, and remain available for later resumption.
Use `plan_only=True` or `--plan-only` to inspect the required chunks and transfer
estimate before exporting. Planning reads HDF5 metadata, so it still uses the network. `expected_size` / `--expected-size` can additionally check an
exact source byte count from the catalog. The cache verifies stored payloads
before reuse; remote resource changes invalidate stale bytes. Use separate
output/cache locations for concurrent independent calls. Source provenance
may include a stable public URL, but credentials and signed query URLs are never
persisted.

## Stack GSLCs into a time series

```python
from nisar_pytools import stack_gslcs

stack = stack_gslcs(
    ["gslc_date1.h5", "gslc_date2.h5", "gslc_date3.h5"],
    frequency="frequencyA",
    polarization="HH",
)
# Sorted by time, grid-validated, dask-backed, CRS assigned
```

## Processing

The [processing reference](docs/processing.md) covers building products rather
than reading them: interferogram formation, coherence, multilooking,
unwrapping, the production RSLC to GUNW pipeline, dolphin prep, phase linking,
polarimetric decomposition, and local incidence angle.

## Roadmap

- [x] Lazy HDF5 reader returning xarray DataTree with CRS
- [x] Prep dolphin to run GSLC to dolphin ready yaml + geotiffs
- [x] ASF search and parallel download with validation
- [x] GSLC time-series stacking
- [x] Interferogram, coherence, multilooking, phase extraction
- [x] Phase unwrapping (SNAPHU)
- [x] Phase linking (EMI with SHP selection)
- [x] Polarimetric decomposition (H-A-alpha)
- [x] Local incidence angle computation
- [x] Visualization helpers (amplitude, phase, interferogram, coherence)
- [ ] Support for additional NISAR product types

## Contributing

Contributions are welcome! Please fork the repo and open a pull request, or open an issue to suggest improvements.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
