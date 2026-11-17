# Data generation

Scripts that generate the `pixelset/` dataset: a site-indexed collection of
daily meteorological forcing, leaf area index (LAI), and plant functional type
(PFT) composition for **1992–2019**, built for training the PhenoNN.

---

## Coordinate contract

The 0.05° `selected_pixels005_*.nc` file defines the **target** sites. LAI, PFT,
daily LAI, and the normalisation target statistics all use the same
`pix_LLLL_OOOOO` IDs on the native 0.05° grid.

The report route stores ERA5 features once per **0.1°** cell, using
`E{lat_idx}_{lon_idx}` IDs. `make_selected_pixels_01.py` writes
`selected_pixels01_1.nc`, which carries both the deduplicated 0.1° cells and a
`parent_site` mapping from every 0.05° target site to its containing 0.1° cell.
Training and prediction pass that file as `--parent_map`; the loader broadcasts
the parent ERA5 values back onto the 0.05° targets. The 0.05° identity route,
where features and targets share `pix_...` IDs, remains supported.

---

## Two resolutions

| Data | Native resolution | Grid shape | Longitude convention | Role |
|---|---|---|---|---|
| **LAI, PFT** | **0.05°** (~5.5 km) | 3600 × 7200 | −180° … 180° | Site grid (identity, targets) |
| **ERA5-Land meteo** | **0.1°** (~11 km) | 1801 × 3600 | 0° … 360° | Forcing (features) |

One 0.1° cell contains up to four 0.05° cells. The target grid is the **0.05°**
LAI/PFT grid; on the report route each 0.05° site inherits the meteo of its
containing 0.1° ERA5 cell, so neighbouring 0.05° sites share identical ERA5
values while each keeps its own LAI and PFT. The 0.05° ERA5 extractor remains
available for the legacy identity route.

---

## The chain

```text
CO2_1700_2023_TRENDYv2024.txt (external input)
│
valid_pixels_{YYYY}.nc (per-year 0.05° validity masks, from compute_valid_pixels.py)
│
▼
select_pixels.py  ──►  all_pixels_selected.nc         (--fraction 1.0, all land pixels)
│                      selected_pixels005_1.nc        (--fraction ~0.1, valid subset)
│
▼
rebalance_pixels.py     (PFT-dominance route)     ──► selected_pixels005_balanced.nc
  or
rebalance_spatial.py    (10° block-density route) ──► selected_pixels005_balanced.nc
                                                    + rebalance_blocks.csv
│
▼
make_selected_pixels_01.py
  selected_pixels005_1.nc (superset of balanced) ──► selected_pixels01_1.nc
│
▼
build_daily_dataset_era5idx.py   (0.1° report route)
  ERA5-Land hourly ──► era5_pixelset/ERA5_daily_pixelset_Y.nc   (10 base vars)
  or
build_daily_dataset_pixelset.py  (0.05° legacy identity route)
  ERA5-Land hourly ──► era5_pixelset/ERA5_daily_pixelset_Y.nc
│
▼
build_pixelset_targets.py
  full LAI grid    ──► LAI_pixelset/LAI_dekadal_Y.nc
  full PFT grid    ──► PFT_pixelset/PFTmap_Y.nc
│
▼
interpolate_lai_daily_linear.py
  LAI_pixelset     ──► LAI_daily_linear/LAI_daily_Y.nc
│
▼
add_daylength.py + add_soil_moisture_proxy.py
  era5_pixelset    ──► same files + daylength + SMI  (12 dynamic vars)
  (optional) add_pheno_thermal.py ──► era5_pheno/
│
▼
create_stats_file.py
  era5 + LAI + PFT + CO2 ──► norm_stats_1992_2019.json
│
▼
(training pipeline consumes era5_pixelset/ + LAI_daily_linear/ +
 norm_stats.json + CO2 LUT + selected_pixels01_1.nc; PFT_pixelset/ for PFT
 conditioning)
```

---

## Step 0 — The one external input

**`CO2_1700_2023_TRENDYv2024.txt`**

- An external CO₂ forcing table in `Annee_YYYY=VALUE` format.
- Read **once**, at Step 9, by `create_stats_file.py --co2_path`.

---

## Step 1 — Build the validity masks

**Script:** `compute_valid_pixels.py`

**Inputs:**
- Full-grid `LAI_dekadal_{Y}.nc`
- Full-grid `PFTmap_{Y}.nc`

**What it does:** for each year, evaluates the LAI/PFT validity criteria cell
by cell on the native 0.05° grid and writes a boolean mask. The masks are the
raw material from which the site set is sampled. For the report dataset the
criteria are PFT sum > 0.99, bare soil PFT1 < 0.4, and at least 34 valid LAI
dekads out of 36. Pass these explicitly; the script defaults are more
permissive.

**Output:** `valid_pixels_{YYYY}.nc` — per-year 0.05° boolean masks.

> This script is **only needed if you want to rebuild the validity masks** from
> the raw LAI/PFT grids. If you already have the selection files (Step 2), you
> can skip it.

---

## Step 2 — Sample the site set

**Script:** `select_pixels.py`

**Inputs:**
- A directory of `valid_pixels_{YYYY}.nc` (from Step 1)

**What it does:**
1. Opens every `valid_pixels_{YYYY}.nc` and counts, per pixel, how many years
   it was valid.
2. Keeps pixels with `counter >= --min_years` (default `1` = union of all years;
   pass `N` = number of files for the intersection).
3. Samples `--fraction` of the qualifying pixels **without replacement**, with
   a seeded RNG (`--seed`, default `42`).
4. Writes a single selection file with `site_id = "pix_LLLL_OOOOO"`
   (4-digit latitude index, 5-digit longitude index).

**Outputs (depending on the invocation):**

| File | Invocation | Meaning |
|---|---|---|
| `all_pixels_selected.nc` | `--fraction 1.0 --min_years 1` | **All** land pixels valid ≥1 year, the reference distribution |
| `selected_pixels005_1.nc` | `--fraction ~0.1 --min_years 1` on the 0.05° masks | 0.05° valid subset used by targets and the parent map |
| `selected_pixels01_1.nc` | `make_selected_pixels_01.py` on `selected_pixels005_1.nc` | 0.1° deduplicated cells plus the 0.05° to 0.1° parent map |

**Output schema:** `site = N_selected`; `lat_idx` (int32), `lon_idx` (int32),
`latitude` (float32), `longitude` (float32), `site_id` (str).
Attrs: `fraction`, `seed`, `min_years`, `n_qualifying`, `n_total`, `n_selected`,
`source`, `valid_files`.

---

## Step 3 — Rebalance the site set

Two rebalancing routes exist. They produce the **same output filename**
(`selected_pixels005_balanced.nc`) but implement **different corrections**. Only
one of them was used for a given dataset; the presence of `rebalance_blocks.csv`
indicates the spatial route.

### Step 3a — PFT-dominance rebalancing

**Script:** `rebalance_pixels.py`

**Inputs:**
- `selected_pixels005_1.nc` (the 0.05° valid subset)
- `PFT_pixelset/PFTmap_2010.nc` (one PFT year; PFT is ~static)

**What it does:**
1. For each site, compute the dominant PFT = argmax of the 15 PFT fractions
   from the single PFT year.
2. Count sites per dominant PFT (the "before" histogram). Tropical broadleaf
   evergreen (PFT 2) is over-represented: ~34 % of pure cells versus ~14 % over
   all land.
3. Decide how many reducible sites (those whose dominant PFT is in
   `--reduce_pft`, default PFT 2) to keep, from either:
   - `--target_total N` → keep roughly N sites in total, or
   - `--target_share 0.15` → reduce PFT 2 until it is 15 % of the pool.
4. Draw a **uniform random subset without replacement** of the reducible sites
   (`--seed 42`, so the draw is exactly reproducible). Non-reducible sites are
   always kept.
5. Write the reduced list.

**Output:** `selected_pixels005_balanced.nc` — same schema as the input, just
fewer rows.

**What is *not* rebuilt:** the underlying ERA5, LAI, and PFT pixel sets. They
were built on the full valid set; the balanced set is a subset of it, so
training just loads fewer sites from the same files. The unreduced 0.1° parent
map remains a superset and can still be used as `--parent_map` during training.

### Step 3b — Spatial (10° block-density) rebalancing

**Script:** `rebalance_spatial.py`

**Inputs:**
- `all_pixels_selected.nc` (the reference distribution — all land pixels)
- `selected_pixels005_1.nc` (the target — valid subset)
- `--block_deg 10` (block size in degrees)

**What it does:** matches the per-block density of the valid subset to the
per-block density of the all-land reference, so the kept sites are not
concentrated in a few 10° blocks. Draws a seeded random subset within each
block to hit `--target_total`.

**Outputs:**
- `selected_pixels005_balanced.nc`
- `rebalance_blocks.csv` — per-block report (before/after counts)

> **Which route was used?** Check for `rebalance_blocks.csv` next to
> `selected_pixels005_balanced.nc`. If it exists → spatial route (3b). If not
> → PFT route (3a).

> **Start point for the data build is therefore:** the selection files from
> Step 2 (or the balanced variant from Step 3) + the raw ERA5-Land hourly files
> + the full-grid LAI and PFT files. The `selected_pixels*.nc` files are the
> coordinate contract.

---

## Step 4 — Deduplicate ERA5 cells and build the parent map

**Script:** `make_selected_pixels_01.py`

**Inputs:** `selected_pixels005_1.nc`, the full 0.05° valid subset (not the
balanced subset, so the map remains a superset).

**What it does:** maps each 0.05° cell to its containing 0.1° ERA5-Land cell,
deduplicates the 0.1° cells, and writes both resolutions to one file:

- `site`: unique 0.1° cells, with raw ERA5 `lat_idx` / `lon_idx` and `E...` IDs;
- `site05`: the original 0.05° IDs and `parent_site`, an index into `site`.

**Output:** `selected_pixels01_1.nc`.

---

## Step 5 — ERA5 daily pixelset (meteo features)

The report route uses `build_daily_dataset_era5idx.py`. The legacy identity
route uses `build_daily_dataset_pixelset.py`; both write the same filenames and
the same base variables.

**Report route script:** `build_daily_dataset_era5idx.py`

**Inputs:**
- `/bdd/ERA5-Land/NETCDF/GLOBAL_01/hourly/FC_SF`, raw ERA5-Land hourly
  NetCDFs (`t2m, d2m, sp, ssrd, strd, tp`)
- `selected_pixels01_1.nc` from `make_selected_pixels_01.py`

**What it does:** extracts ERA5 exactly once per unique 0.1° cell using the raw
ERA5 indices, resamples hourly to daily, derives `VPD_max`, `VPD_mean`,
`Rn_tot`, and `PET`, and writes `E...` site IDs. There is no 0.05° duplication
and no longitude roll.

**Legacy route script:** `build_daily_dataset_pixelset.py`

**Inputs:**
- The same raw ERA5-Land hourly archive
- `selected_pixels005_*.nc`

**What it does:** maps each 0.05° site to its containing 0.1° cell, extracts and
resamples the meteo, then writes native 0.05° `pix_...` IDs. Four neighbouring
sites share the same ERA5 values but are stored four times.

**Output for either route:** `era5_pixelset/ERA5_daily_pixelset_{1992…2019}.nc`

Schema: `time=365, site=N`; base vars `Tmin, Tmax, Tmean, ssrd_sum, strd_sum,
tp_sum, VPD_max, VPD_mean, Rn_tot, PET`. Step 8 appends `daylength` and `SMI`,
completing the 12 dynamic channels.

---

## Step 6 — LAI and PFT pixelsets (targets)

**Script:** `build_pixelset_targets.py`

**Inputs:**
- Full-grid `LAI_dekadal_{Y}.nc` (3600×7200) — `--target_dir`
- Full-grid `PFTmap_{Y}.nc` — `--pft_dir`
- `selected_pixels*.nc` (`lat_idx`, `lon_idx`, `site_id`, `latitude`, `longitude`)

**What it does:** one sequential decompression pass of each full grid, then a
NumPy fancy-index `[lat_idx, lon_idx]` → `(dekad, site)` for LAI and
`(pft, site)` for PFT. No scattered `isel` (that thrashes the chunk cache over
NFS). Writes the site coords onto the output.

**Outputs:**
- `LAI_pixelset/LAI_dekadal_{1992…2019}.nc` — `LAI(dekad=36, site)`
- `PFT_pixelset/PFTmap_{1992…2019}.nc` — `pft_frac(pft=15, site)`

---

## Step 7 — Daily LAI by linear interpolation

**Script:** `interpolate_lai_daily_linear.py`

**Input:** `LAI_pixelset/LAI_dekadal_{Y}.nc`

**What it does:** computes the 36 dekad day-of-year anchors (days 5/15/25,
leap-safe), then for each site linearly interpolates to 365 daily values with
`np.interp`. Needs ≥2 finite dekads; otherwise NaN. Edge values are held
constant (no extrapolation).

**Output:** `LAI_daily_linear/LAI_daily_{1992…2019}.nc` — `LAI(day=365, site)`,
carrying the same site coords.

---

## Step 8 — Augment the ERA5 features

The report contract needs 12 dynamic features. Step 5 writes the 10 base
variables; two resolution-agnostic scripts append the remaining channels:

- `add_daylength.py` reads `era5_pixelset/ERA5_daily_pixelset_{Y}.nc`, adds
  `daylength(time, site)` from latitude and day-of-year (FAO-56 declination),
  and writes the same layout to `--out_dir`. `--out_dir` may equal
  `--features_dir` for an in-place update.
- `add_soil_moisture_proxy.py` reads the same files, adds `SMI(time, site)`
  from a causal 30-day precipitation weighted sum, and preserves the source
  chunking. It also supports in-place updates.

The two passes are independent. Running both leaves `era5_pixelset/` with
`Tmin ... PET, daylength, SMI`.

`add_pheno_thermal.py` is optional and off by default. It adds 10 ORCHIDEE
thermal phenology features and writes `era5_pheno/`; do not enable it for the
report route.

---

## Step 9 — Normalisation statistics

**Script:** `create_stats_file.py`

**Inputs:**
- `era5_pixelset/ERA5_daily_pixelset_{Y}.nc` — `--features_dir`
- `LAI_pixelset/LAI_dekadal_{Y}.nc` — `--target_dir`
- `PFT_pixelset/PFTmap_{Y}.nc` — `--pft_dir`
- `CO2_1700_2023_TRENDYv2024.txt` — `--co2_path`

**What it does:** scans every year, accumulating online mean/std
(`RunningStats`) for:
- `DYNAMIC_FEATURES` (METEO [+ DERIVED]) — with `log1p` applied first for
  `LOG_TRANSFORM_FEATURES`
- `PFT_COLS` (`pft1_frac..pft15_frac`, pooled over years)
- `LAI` (target)
- `co2` (from the TRENDY LUT, per year in range)

Optional `--n_sites` subsamples sites per file for speed. Output is
byte-identical to the previous format.

**Output:** `norm_stats_1992_2019.json` — `{feat: {mean, std}}` for every
channel, plus an optional `_transforms` block selecting `zscore`,
`log1p_zscore`, `sqrt_zscore`, `cbrt_zscore`, `asinh_zscore`, `minmax` or
`none` per channel.

**Profiles:** `--profile {default,n1,n2,n3,p_sqrt,p_cbrt,p_asinh,p_minmax}` emits the
named experiment policies (see `phenonn/datagen/norm_profiles.py`);
`--transform_spec file.json` overrides them channel by channel. `default`
writes the same file as before.

---

## Step 10 — Diagnostics (not part of the data build)

**Script:** `diagnose_selected_pixels.py` (and `_bis`, `_pft`)

**Inputs:** `selected_pixels*.nc`, `PFT_pixelset/PFTmap_{1992…2019}.nc`

**What it does:** averages PFT fractions over the year range (joined by
`site_id`), renders world maps (dominant PFT, pure cells above threshold,
density per 10° block) and a purity table. Uses one throwaway subprocess per
`.nc` to avoid HDF5 heap corruption on `.venv_ERA5`.

**Outputs (in `analysis/`):**
- `selected_pixels_diag_dominant_pft.png`
- `selected_pixels_diag_pure_gt60.png`
- `selected_pixels_diag_purity_table.{csv,png}`
- `selected_pixels_diag_density_10deg.png`

`pft_diagnostics.py` is a *different* diagnostic: it reads `LAI_pixelset` +
`era5_pixelset` for a single-PFT subset (`selected_pixels_PFT{k}.nc`) and
writes `lai_seasonal.png`, `features_hist.png`, `features_dist_by_year.png`,
`features_seasonal.png`. **These filenames are not in `analysis/`** — so
`analysis/` was produced by `diagnose_selected_pixels*.py`, not
`pft_diagnostics.py`.

---

## The scripts — role in this chain

| File | Step | Produces |
|---|---|---|
| `compute_valid_pixels.py` | 1 | `valid_pixels_{YYYY}.nc` (per-year masks) |
| `select_pixels.py` | 2 | `all_pixels_selected.nc`, `selected_pixels005_1.nc` |
| `rebalance_pixels.py` | 3a | `selected_pixels005_balanced.nc` (PFT route) |
| `rebalance_spatial.py` | 3b | `selected_pixels005_balanced.nc` + `rebalance_blocks.csv` (spatial route) |
| `make_selected_pixels_01.py` | 4 | `selected_pixels01_1.nc` (0.1° cells + parent map) |
| `build_daily_dataset_era5idx.py` | 5 (report) | `era5_pixelset/ERA5_daily_pixelset_Y.nc` (0.1° route) |
| `build_daily_dataset_pixelset.py` | 5 (legacy) | `era5_pixelset/ERA5_daily_pixelset_Y.nc` (0.05° route) |
| `build_pixelset_targets.py` | 6 | `LAI_pixelset/LAI_dekadal_Y.nc`, `PFT_pixelset/PFTmap_Y.nc` |
| `interpolate_lai_daily_linear.py` | 7 | `LAI_daily_linear/LAI_daily_Y.nc` |
| `add_daylength.py` | 8 | `daylength` appended to `era5_pixelset/` |
| `add_soil_moisture_proxy.py` | 8 | `SMI` appended to `era5_pixelset/` |
| `add_pheno_thermal.py` | 8 (optional) | `era5_pheno/` |
| `create_stats_file.py` | 9 | `norm_stats_1992_2019.json` |
| `diagnose_selected_pixels.py` / `_bis.py` | 10 | `analysis/selected_pixels_diag_*` |
| `pft_diagnostics.py` | 10 (separate) | `lai_seasonal.png`, `features_*.png` (not in `analysis/`) |

---

## Reproducibility checklist

To rebuild `pixelset/` from scratch you need:

- [ ] The raw LAI and PFT full grids (`LAI_dekadal_{Y}.nc`, `PFTmap_{Y}.nc`)
- [ ] The raw ERA5-Land hourly archive
      (`/bdd/ERA5-Land/NETCDF/GLOBAL_01/hourly/FC_SF`)
- [ ] `CO2_1700_2023_TRENDYv2024.txt`
- [ ] The scripts listed in the table above

Run order:

```
Step 1  compute_valid_pixels.py       -> valid_pixels_{YYYY}.nc
Step 2  select_pixels.py              -> selected_pixels005_1.nc
                                         (optionally all_pixels_selected.nc)
Step 3  rebalance_pixels.py  (or)
        rebalance_spatial.py          -> selected_pixels005_balanced.nc
Step 4  make_selected_pixels_01.py    -> selected_pixels01_1.nc
Step 5  build_daily_dataset_era5idx.py (report)  -+
        build_daily_dataset_pixelset.py (legacy)   | (5 and 6 can run in parallel)
Step 6  build_pixelset_targets.py                 -+
Step 7  interpolate_lai_daily_linear.py   <- depends on Step 6
Step 8  add_daylength.py
        add_soil_moisture_proxy.py        <- depends on Step 5
Step 9  create_stats_file.py              <- depends on Steps 5, 6, 8, and CO2
Step 10 diagnose_selected_pixels.py       (diagnostic only)
```

For the report route, train and predict with
`--selected_pixels selected_pixels005_balanced.nc` and
`--parent_map selected_pixels01_1.nc`. The parent map is built from the full
`selected_pixels005_1.nc` pool, so it remains a superset of the balanced subset.

If you already have the selection files from Step 2 (and optionally the
balanced variant from Step 3), you can start at Step 4.
