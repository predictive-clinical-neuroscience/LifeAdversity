# LifeAdversity

Scripts to run the analysis detailed our manuscript:

**A neurobiological signature of childhood trauma predicts mental illness in adults**

Linda Schlüter, Tanja M. Brückl, Hannah S. Savage, Victor I. Spoormaker, Philipp G. Sämann, Nilakshi Vaidya, Michael Czisch, Jasper van Oort, Janna N. Vrijsen, Philip F.P. van Eijndhoven, Lauren Robinson, Jeanne Winterer, Sinead King, Hedi Kebir, Hervé Lemaître, Ulrike Schmidt, Julia Sinclair, Argyris Stringaris, Marina Bobou, Zuo Zhang, PhD10,18; Gareth J. Barker, Arun L.W. Bokde, Rüdiger Brühl, Herta Flor, Hugh Garavan, Penny Gowland, Antoine Grigis, Andreas Heinz, Jean-Luc Martinot, Marie-Laure Paillère Martinot, Eric Artiges, Frauke Nees, Dimitri Papadopoulos Orfanos, Tomáš Paus, Luise Poustka, Michael N. Smolka, Robert Whelan, Paul Wirsching, Tobias Banaschewski, Sylvane Desrivières, Indira Tendolkar, Henrik Walter, Elisabeth B. Binder, Gunter Schumann, Healthy Brain Study Consortium, BeCOME Working Group, IMAGEN Consortium, ESTRA & STRATIFY consortium, environMENTAL consortium, Peter C.R. Mulders, Nathalie E. Holz & Andre F. Marquand

---

### 1. Data preparation:
  - `01_imputation_ct_mindset` applies MICE (Multiple Imputation by Chained Equations) to impute missing childhood trauma scores
  - `02_rescale_ct_combined_cohorts` applies min-max scaling to the childhood trauma scores

### 2. Normative models:
  - `01_prepare_wholebrain_model.py`
  - `01_prepare_wholebrain_model_reverse.py` reverses the train-test assignment of subjects
  - `02_run_wholebrain_model.py`
  - `03_evaluate_wholebrain_model.py`
  - `04_compute_structure_coefficients.py`

### 3. Exploratory factor analysis:
  - `01_efa_symptom_domains`

### 4. Canonical correlation analysis:
  - `02_run_scca_enet.py`
  - `03_scca_plot_loadings.py`
  - `04_scca_permutation.py`
