# ShadowProject

ShadowProject is a preprocessing scaffold for NOAA-guided and spherical-Gaussian-conditioned street-view shadow generation. The long-term research goal is to generate physically plausible object shadows for street-view object insertion by combining solar/light priors, coarse shadow sketches, and later diffusion-based refinement.

## Scaffold Scope

This v1 scaffold only implements deterministic preprocessing utilities:

- DESOBAv2 dataset inspection
- pseudo light direction estimation from object and shadow masks
- K=2 spherical Gaussian lighting parameter generation
- coarse physics-guided shadow sketch generation

This stage does not include OpenWeatherMap, MoGe-2, SAM, ControlNet training, SGDiffusion training, or any external weather/geometry/segmentation integration.

## DESOBAv2 Layout

Place DESOBAv2 files under:

```text
data/desobav2/
├── composite/
├── target/
├── object_mask/
└── shadow_mask/
```

The expected subdirectories can be changed in `configs/dataset_config.yaml`. Samples are matched by filename stem, so `composite/0001.png`, `target/0001.png`, `object_mask/0001.png`, and `shadow_mask/0001.png` form one sample.

DESOBAv2 does not provide reliable GPS or timestamp metadata for every sample. This scaffold therefore uses pseudo light directions estimated from masks and does not label those directions as NOAA solar directions.

## Pipeline

Run the preprocessing steps from the repository root:

```powershell
python scripts/inspect_dataset.py --config configs/dataset_config.yaml
python scripts/generate_pseudo_light.py --config configs/dataset_config.yaml
python scripts/generate_sg_params.py --config configs/dataset_config.yaml
python scripts/generate_shadow_sketch.py --config configs/dataset_config.yaml
```

Outputs are written to:

```text
data/intermediate/pseudo_light/
data/intermediate/sg_params/
data/intermediate/shadow_sketch/
```

## Tests

Run all tests:

```powershell
python -m unittest discover tests
```

Run the existing light-engine tests only:

```powershell
python -m unittest tests.test_light_engine
```

