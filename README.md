# SUCNR1 Metabolic Flexibility Model

This repository provides an end‐to‐end Python implementation of a
phenomenological, SBML‐based ordinary differential equation (ODE)
model of hepatic metabolic flexibility downstream of the succinate
receptor 1 (SUCNR1). The primary objective is to recapitulate
glucose, insulin and pyruvate tolerance tests and
fasting/refeeding dynamics observed in the accompanying study
“SUCNR1 coordinates metabolic flux, mitochondrial function, and
nutrient‐dependent adaptation in hepatocytes. The model
couples a simple plasma compartment to a minimal hepatic module and
allows calibration of rate constants to experimental time–series data.

> **Disclaimer:** This model is intentionally compact. It is not
> intended to replace genome–scale reconstructions or high‐fidelity
> metabolic models. Instead it provides a reproducible starting
> point for exploring SUCNR1‐dependent phenomena, using the data
> contained in the supplementary material of the paper. Many
> kinetic and structural choices are phenomenological and should be
> revisited as new data become available.

--- 

## Scientific aim

Succinate is both an intermediate of the tricarboxylic acid (TCA)
cycle and an extracellular signal sensed by the G protein–coupled
receptor SUCNR1. Marsal‑Beltran et al. show that hepatic succinate
and Sucnr1 expression rise after feeding and that hepatocyte‐specific
knockout of Sucnr1 promotes a fasting‐like phenotype with enhanced
gluconeogenesis and altered mitochondrial fluxes【9†L9-L11】. The
objective of this package is to translate these qualitative
observations into a low dimensional ODE system whose parameters can
be estimated from tolerance tests, fasting glucose/ketones and
Seahorse flux assays. Once calibrated, the model can be perturbed
to explore in silico knockouts and pharmacological interventions.

Reference:

Anna Marsal-Beltran et al. ,SUCNR1 coordinates metabolic flux, mitochondrial function, and nutrient-dependent adaptation
in hepatocytes.Sci. Adv.12,eaec8873(2026).DOI:10.1126/sciadv.aec8873

## Installation

Use [uv](https://github.com/astral-sh/uv) (or `pip`/`virtualenv`) to
install the package and its dependencies. Python 3.11 or higher is
required.

```bash
git clone https://github.com/bibymaths/sucnr1-metaflex.git
cd sucnr1-metaflex
uv pip install .[dev]
```

--- 

## Step-wise workflow

The commands below assume that the package is installed in the active environment and that the repository is being run
from the project root.

```bash
uv pip install -e ".[dev]"
```

or, with an already active virtual environment:

```bash
pip install -e ".[dev]"
```

### 1. Ingest and clean the supplementary data

Inspect the supplementary archive:

```bash
sucnr1-data inventory \
  --zip data/Beltran2026_Supp.zip \
  --out results
```

Parse the selected dynamic assays into tidy CSV files:

```bash
sucnr1-data ingest \
  --zip data/Beltran2026_Supp.zip \
  --out results/processed
```

After ingestion, clean assay time units and remove AUC pseudo-time rows:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("results/processed")

body = pd.read_csv(p / "dynamic_body.csv")
sea = pd.read_csv(p / "dynamic_seahorse.csv")

body["assay"] = body["assay"].astype(str)
sea["assay"] = sea["assay"].astype(str)

minute_assays = {
    "GTT", "GTT_female",
    "ITT", "ITT_female",
    "PTT", "PTT_female",
}

fasting_assays = {
    "fasting_glucose",
    "fasting_ketone",
}

body_parts = []

for assay, df in body.groupby("assay", dropna=False):
    df = df.copy()

    if assay in minute_assays:
        df = df[df["time"] <= 120.0]
        df["time"] = df["time"] / 60.0
    elif assay in fasting_assays:
        df = df[df["time"] <= 24.0]
    else:
        df = df[df["time"] <= 120.0]

    body_parts.append(df)

body = pd.concat(body_parts, ignore_index=True)

sea = sea[sea["time"] <= 180.0].copy()
sea["time"] = sea["time"] / 60.0

body.to_csv(p / "dynamic_body.csv", index=False)
sea.to_csv(p / "dynamic_seahorse.csv", index=False)
pd.concat([body, sea], ignore_index=True).to_csv(p / "all_tidy.csv", index=False)

for path in [
    p / "dynamic_body.csv",
    p / "dynamic_seahorse.csv",
    p / "all_tidy.csv",
]:
    df = pd.read_csv(path)
    print("\n", path)
    print(df.groupby("assay")["time"].agg(["min", "max", "nunique"]).to_string())
PY
```

Expected maximum times after cleaning:

```text
GTT / ITT / PTT: approximately 2.0 hours
fasting_glucose / fasting_ketone: 24.0 hours
Seahorse assays: approximately 1.5-2.1 hours
```

### 2. Build SBML models

Build the plasma/body model:

```bash
sucnr1-build body \
  --config configs/model_body.yaml \
  --out results/models/body.xml
```

Build the liver model:

```bash
sucnr1-build liver \
  --config configs/model_liver.yaml \
  --out results/models/liver.xml
```

Build the coupled body-liver model:

```bash
sucnr1-build combined \
  --body-config configs/model_body.yaml \
  --liver-config configs/model_liver.yaml \
  --config configs/model_combined.yaml \
  --out results/models/body_liver.xml
```

If the installed CLI exposes the older combined-model interface, use:

```bash
sucnr1-build combined \
  --config configs/model_combined.yaml \
  --out results/models/body_liver.xml
```

Validate the generated models:

```bash
python - <<'PY'
import libsbml

for path in [
    "results/models/body.xml",
    "results/models/liver.xml",
    "results/models/body_liver.xml",
]:
    print("\n", path)
    doc = libsbml.readSBML(path)
    n = doc.checkConsistency()
    print("messages:", n)

    for i in range(n):
        e = doc.getError(i)
        if e.getSeverity() >= libsbml.LIBSBML_SEV_WARNING:
            print("severity", e.getSeverity(), "-", e.getMessage().splitlines()[0])
PY
```

### 3. Run pre-fit simulations

Before fitting, confirm that the models are dynamic and not structurally flat:

```bash
python scripts/simulate_plot_all_models.py \
  --start 0 \
  --end 24 \
  --num 240 \
  --out results/simulations/prefit
```

This writes one CSV per model and diagnostic plots under:

```text
results/simulations/prefit/
```

### 4. Fit body, liver and combined models

Remove stale fits before recalibration:

```bash
rm -rf results/runs/body_fit results/runs/liver_fit results/runs/combined_fit
```

Fit the body model to body dynamic assays:

```bash
sucnr1-fit \
  --data results/processed \
  --model results/models/body.xml \
  --config configs/fit_body.yaml \
  --out results/runs/body_fit/fit \
  --n-starts 10
```

Fit the liver model to Seahorse assays:

```bash
sucnr1-fit \
  --data results/processed \
  --model results/models/liver.xml \
  --config configs/fit_liver.yaml \
  --out results/runs/liver_fit/fit \
  --n-starts 10
```

Fit the combined model to all processed dynamic assays:

```bash
sucnr1-fit \
  --data results/processed \
  --model results/models/body_liver.xml \
  --config configs/fit_combined.yaml \
  --out results/runs/combined_fit/fit \
  --n-starts 10
```

Inspect ranked multistart losses:

```bash
cat results/runs/body_fit/fit/ranked_multistart.csv
cat results/runs/liver_fit/fit/ranked_multistart.csv
cat results/runs/combined_fit/fit/ranked_multistart.csv
```

Inspect best-fit parameters:

```bash
cat results/runs/body_fit/fit/best_parameters.csv
cat results/runs/liver_fit/fit/best_parameters.csv
cat results/runs/combined_fit/fit/best_parameters.csv
```

### 5. Plot fit diagnostics

Generate body fit plots:

```bash
sucnr1-plot fit \
  --fit-dir results/runs/body_fit/fit \
  --data results/processed \
  --model results/models/body.xml \
  --config configs/fit_body.yaml \
  --out results/figures/body_fit
```

Generate liver fit plots:

```bash
sucnr1-plot fit \
  --fit-dir results/runs/liver_fit/fit \
  --data results/processed \
  --model results/models/liver.xml \
  --config configs/fit_liver.yaml \
  --out results/figures/liver_fit
```

Generate combined fit plots:

```bash
sucnr1-plot fit \
  --fit-dir results/runs/combined_fit/fit \
  --data results/processed \
  --model results/models/body_liver.xml \
  --config configs/fit_combined.yaml \
  --out results/figures/combined_fit
```

List generated figures and residual tables:

```bash
find results/figures \
  -type f \( -name "*.png" -o -name "*.csv" \) \
  | sort
```

### 6. Run post-fit simulations

Simulate all three fitted models and generate plots:

```bash
python scripts/simulate_plot_all_models.py \
  --start 0 \
  --end 24 \
  --num 240 \
  --out results/simulations/postfit
```

By default the script uses these parameter files if they exist:

```text
results/runs/body_fit/fit/best_parameters.json
results/runs/liver_fit/fit/best_parameters.json
results/runs/combined_fit/fit/best_parameters.json
```

If a parameter file is missing, the script falls back to the baseline values embedded in the SBML model.

### 7. Run a single forward simulation from the CLI

Body model:

```bash
sucnr1-sim forward \
  --model results/models/body.xml \
  --params results/runs/body_fit/fit/best_parameters.json \
  --start 0 \
  --end 24 \
  --num 240 \
  --selection G_mgdl,G_plasma,Pyr_plasma,AA_plasma,Ketone_plasma,Succ_plasma \
  --out results/simulations/body_fit
```

Liver model:

```bash
sucnr1-sim forward \
  --model results/models/liver.xml \
  --params results/runs/liver_fit/fit/best_parameters.json \
  --start 0 \
  --end 24 \
  --num 240 \
  --selection G6P_liver,Glycogen_liver,Pyr_liver,Succ_mito,Succ_extra,Mito_capacity,OCR_proxy,ECAR_proxy,mito_OCR_proxy \
  --out results/simulations/liver_fit
```

Combined model:

```bash
sucnr1-sim forward \
  --model results/models/body_liver.xml \
  --params results/runs/combined_fit/fit/best_parameters.json \
  --start 0 \
  --end 24 \
  --num 240 \
  --selection G_mgdl,G_plasma,Pyr_plasma,AA_plasma,Ketone_plasma,Succ_plasma,G6P_liver,Glycogen_liver,Pyr_liver,Succ_mito,Succ_extra,Mito_capacity,OCR_proxy,ECAR_proxy,mito_OCR_proxy \
  --out results/simulations/combined_fit
```

### 8. Run scenario simulations

The scenario engine applies multiplicative parameter perturbations from `configs/scenarios.yaml`.

Create a run directory layout expected by the scenario command:

```bash
mkdir -p results/runs/combined_fit/models
cp results/models/body_liver.xml results/runs/combined_fit/models/body_liver.xml
```

Run scenarios with the combined fitted model:

```bash
sucnr1-scenarios \
  --run-dir results/runs/combined_fit \
  --config configs/scenarios.yaml \
  --out results/scenarios/combined_fit
```

If the installed CLI exposes a `run` subcommand, use:

```bash
sucnr1-scenarios run \
  --run results/runs/combined_fit \
  --config configs/scenarios.yaml \
  --out results/scenarios/combined_fit
```

Inspect scenario outputs:

```bash
find results/scenarios/combined_fit \
  -type f \( -name "*.csv" -o -name "*.png" \) \
  | sort
```

### 9. Protocol-driven transient fitting

Calibration now applies assay-specific protocol inputs before each assay/condition simulation. GTT raises the `G_abs`
absorption pool, PTT raises the `Pyr_abs` absorption pool, ITT raises `I_eff`, and fasting assays keep these protocol
pools at zero. Seahorse KO, siRNA, antagonist, and agonist effects are applied as explicit condition factors and
OCR/ECAR protocol shapes rather than autonomous ODE oscillations.

--- 

## License

This project is provided under the MIT license. See the `LICENSE`
file for details.

---

## Protocol-driven calibration model

The basal SUCNR1 models are intentionally compact and stable. Experimental transients are represented by assay protocols
rather than by autonomous oscillatory ODE feedback. For body assays, GTT uses a decaying `G_abs` input pool, PTT uses a
decaying `Pyr_abs` input pool, and ITT uses the existing decaying `I_eff` insulin-action pulse. Fasting protocols set
all protocol input pools to zero.

Seahorse OCR/ECAR peaks and drops are observation-layer protocol shapes (`seahorse_ocr` and `seahorse_ecar`) multiplied
onto stable model observables with explicit genotype/treatment factors. This keeps OCR/ECAR fitting interpretable and
avoids unconstrained succinate-driven oscillations.

Typical commands:

```bash
sucnr1-build body --config configs/model_body.yaml --out results/models/body.xml
sucnr1-build liver --config configs/model_liver.yaml --out results/models/liver.xml
sucnr1-build combined --body-config configs/model_body.yaml --liver-config configs/model_liver.yaml --config configs/model_combined.yaml --out results/models/body_liver.xml
python scripts/simulate_plot_all_models.py --no-fit-params --start 0 --end 2 --num 200 --out results/simulations/prefit_0_2h
sucnr1-fit --data results/processed --model results/models/body.xml --config configs/fit_body.yaml --out results/runs/body_fit/fit --n-starts 3
sucnr1-fit --data results/processed --model results/models/liver.xml --config configs/fit_liver.yaml --out results/runs/liver_fit/fit --n-starts 3
sucnr1-plot fit --fit-dir results/runs/body_fit/fit --data results/processed --model results/models/body.xml --config configs/fit_body.yaml --out results/figures/body_fit
sucnr1-plot fit --fit-dir results/runs/liver_fit/fit --data results/processed --model results/models/liver.xml --config configs/fit_liver.yaml --out results/figures/liver_fit
```
