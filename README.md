# SUCNR1 Metabolic Flexibility Model

This repository provides an end‐to‐end Python implementation of a
phenomenological, SBML‐based ordinary differential equation (ODE)
model of hepatic metabolic flexibility downstream of the succinate
receptor 1 (SUCNR1).  The primary objective is to recapitulate
glucose, insulin and pyruvate tolerance tests and
fasting/refeeding dynamics observed in the accompanying study
“SUCNR1 coordinates metabolic flux, mitochondrial function, and
nutrient‐dependent adaptation in hepatocytes.  The model
couples a simple plasma compartment to a minimal hepatic module and
allows calibration of rate constants to experimental time–series data.

> **Disclaimer:** This model is intentionally compact.  It is not
> intended to replace genome–scale reconstructions or high‐fidelity
> metabolic models.  Instead it provides a reproducible starting
> point for exploring SUCNR1‐dependent phenomena, using the data
> contained in the supplementary material of the paper.  Many
> kinetic and structural choices are phenomenological and should be
> revisited as new data become available.

## Scientific aim

Succinate is both an intermediate of the tricarboxylic acid (TCA)
cycle and an extracellular signal sensed by the G protein–coupled
receptor SUCNR1.  Marsal‑Beltran et al. show that hepatic succinate
and Sucnr1 expression rise after feeding and that hepatocyte‐specific
knockout of Sucnr1 promotes a fasting‐like phenotype with enhanced
gluconeogenesis and altered mitochondrial fluxes【9†L9-L11】.  The
objective of this package is to translate these qualitative
observations into a low dimensional ODE system whose parameters can
be estimated from tolerance tests, fasting glucose/ketones and
Seahorse flux assays.  Once calibrated, the model can be perturbed
to explore in silico knockouts and pharmacological interventions.

## Installation

Use [uv](https://github.com/astral-sh/uv) (or `pip`/`virtualenv`) to
install the package and its dependencies.  Python 3.11 or higher is
required.

```bash
git clone <repository>
cd sucnr1-metaflex
uv pip install .[dev]
```

Alternatively, install the core requirements:

```bash
uv pip install numpy pandas scipy pydantic pyyaml openpyxl typer[all] rich loguru \
  matplotlib plotly dash python-libsbml sbmlutils sbmlsim libroadrunner scikit-learn
```

## Quick start

1. **Inspect and ingest the data:**

   ```bash
   sucnr1-data inventory --zip data/Beltran2026_Supp.zip --out results
   sucnr1-data ingest --zip data/Beltran2026_Supp.zip --out results/processed
   ```

   The `inventory` command reports the names of all workbooks and
   sheets in the zip archive.  The `ingest` command extracts the
   spreadsheets and converts selected dynamic assays to a tidy format
   with time, replicate and genotype information.  Results are
   written to CSV files in the `results/processed` directory.

2. **Build the SBML models:**

   ```bash
   sucnr1-build body --config configs/model_body.yaml --out results/models/body.xml
   sucnr1-build liver --config configs/model_liver.yaml --out results/models/liver.xml
   sucnr1-build combined --config configs/model_combined.yaml --out results/models/body_liver.xml
   ```

   These commands generate SBML Level 3 version 2 documents that
   describe the plasma, hepatic and coupled systems.  All models
   declare explicit units, compartments and state variables.  After
   generation, the code validates the XML with `libsbml` and reports
   any inconsistencies.

3. **Fit the model to data:**

   ```bash
   sucnr1-fit run --data results/processed --config configs/fit.yaml --out results/runs --n-starts 10
   ```

   A basic least–squares objective is formulated from the glucose
   tolerance test (GTT), insulin tolerance test (ITT), pyruvate
   tolerance test (PTT), fasting glucose and fasting ketone data.  The
   optimiser performs multiple random initialisations (`--n-starts`)
   and ranks solutions by total loss.  Best‐fitting parameters are
   stored as CSV/JSON alongside residual plots.

4. **Simulate scenarios:**

   ```bash
   sucnr1-sim steady-state --model results/runs/<run>/models/body_liver.xml \
     --params results/runs/<run>/fit/best_parameters.json
   sucnr1-sim forward --model results/runs/<run>/models/body_liver.xml \
     --params results/runs/<run>/fit/best_parameters.json --protocol fasting
   sucnr1-scenarios run --run results/runs/<run> --config configs/scenarios.yaml
   ```

   Steady–state and forward simulations allow inspection of model
   behaviour outside of the experimental time points.  The scenario
   engine applies multiplicative perturbations to reaction rates or
   initial conditions to emulate gene knockouts or pharmacological
   inhibition.  Results are saved as CSV and PNG.

5. **Generate reports and launch the dashboard:**

   ```bash
   sucnr1-report build --run results/runs/<run>
   sucnr1-dashboard --run results/runs/<run>
   ```

   The report builder exports markdown documents summarising the
   model structure, calibration metrics and scenario outcomes.  The
   interactive dashboard (built on Dash) lets you overlay
   simulations with experimental means ± SEM, explore parameter
   ensembles and run custom in silico perturbations.

## Model assumptions and limitations

- **Phenomenological rate laws:**  The ODE system captures only the
  main processes required to reproduce the tolerance tests (e.g.,
  glucose appearance, insulin‐dependent clearance, hepatic
  gluconeogenesis and basic ketogenesis).  It is not a detailed
  reconstruction of glycolysis, gluconeogenesis or the full TCA
  cycle.
- **Shared base parameters:**  Wild type and knockout animals share
  most kinetic constants, with genotype‐specific multipliers (e.g.,
  `theta_KO_gng`) used to represent Sucnr1 deletion.  This design
  prevents overfitting separate models to each genotype.
- **Data constraints:**  Only a subset of the supplementary tables
  provide time–series measurements; the remainder are treated as
  qualitative or auxiliary constraints.  The calibration routine
  currently fits to GTT, ITT, PTT, fasting glucose and ketones.  OCR
  and ECAR data are parsed and exposed, but full joint fitting to
  Seahorse traces is left as future work.
- **Identifiability:**  Even simple models can be poorly
  identifiable.  Use the ranked ensemble of solutions rather than
  trusting a single parameter set.  Confidence intervals are
  approximate and do not capture structural uncertainties.
- **Parsing heuristics:**  The data ingestor uses heuristics to
  interpret the complex Excel layouts.  Column names, genotype
  labels and replicate indices are inferred from the header rows.  If
  a sheet deviates from the expected structure, a warning is logged
  and the records are skipped.

## License

This project is provided under the MIT license.  See the `LICENSE`
file for details.