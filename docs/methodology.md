# Research Design — AI-Driven Predictive Simulation of Gait & Post-Treatment Functional Outcomes

*Husain Naqvi · 2026 · All Rights Reserved*

## 1. Problem summary

**Domain:** clinical biomechanics, rehabilitation medicine and neuromusculoskeletal physiology.
**Question:** a patient's gait is measured once when they enter inpatient rehabilitation after a lower-limb injury (hip, knee, ankle or calcaneus). From that one visit, can we **simulate what their gait will look like after N days of treatment**, put an **honest uncertainty band** around it, and **flag who will reach a clinically meaningful improvement**?

**Why it matters biologically:** gait recovery reflects tissue healing, load-bearing tolerance, motor re-learning and pain-avoidance strategies. Each of these leaves a signature in the ground-reaction-force (GRF) waveform, such as an asymmetric loading peak, a shortened push-off or a shifted centre of pressure. A model that forecasts the whole waveform, not just one number, helps clinicians:
- set realistic discharge goals,
- spot non-responders early, so their treatment can be changed,
- run "what-if" checks (more time in rehab, different footwear) before committing resources.

**Why AI:** the input is high-dimensional (10 force/COP channels × 101 time points), the recovery is nonlinear and depends on diagnosis, and the target is itself a curve. Classical regression on a few summary features throws away most of the signal.

**Data:** GaitRec (Horsak et al., 2020, *Scientific Data* 7:143; CC BY 4.0). Gait was recorded with force plates at an Austrian (AUVA) rehabilitation centre: **2,084 patients and 211 healthy controls**. **1,858 patients have repeated sessions inside the same admission**, at a median of about 3 weeks apart, so the data contain real before/after treatment pairs.

## 2. Five candidate AI-biology solutions

| # | Solution | Algorithms | Biological rationale and insight |
|---|---|---|---|
| S1 | **Functional PCA + regularised regression** | PCA, Ridge / Elastic Net | Gait waveforms lie on a low-dimensional manifold (loading, push-off, symmetry modes). Regressing follow-up mode scores on baseline modes gives an interpretable "recovery operator". |
| S2 | **Gradient-boosted recovery model** | HistGradientBoosting / XGBoost on PC scores and clinical covariates | Captures nonlinear interactions such as age × fracture site × time. Feature importance shows which baseline deficits predict slow recovery. |
| S3 | **Probabilistic recovery trajectories** | Gaussian Process Regression over time since admission | Treats recovery as a smooth stochastic process with calibrated uncertainty. Fits naturally with irregular visit timing. |
| S4 | **Deep conditional forecaster** | 1-D CNN encoder + FiLM conditioning on clinical context and elapsed time, deep ensemble | Learns local waveform features (impact transient, mid-stance unloading, push-off) directly. Conditioning on time turns it into a simulator you can query at any day. |
| S5 | **Recovery phenotyping** | VAE latent space + Gaussian Mixture Model | Unsupervised discovery of recovery phenotypes (fast, plateau, compensatory) that may match tissue-level healing differences. |

## 3. Five innovative hybrid methodologies

| # | Hybrid | How it improves on classical methods |
|---|---|---|
| H1 | **CNN encoder + FiLM time conditioning + β-NLL heteroscedastic head + deep ensemble ("GaitTwin")** | One model gives the whole post-treatment waveform for any horizon, with aleatoric (patient variability) and epistemic (model ignorance) uncertainty. Split-conformal calibration then guarantees interval coverage. |
| H2 | **Healthy-reference manifold (PCA) + forecaster → GRF Normalcy Index (GNI)** | Turns a 1,010-number forecast into one clinically readable outcome, scaled like the Gait Deviation Index: 100 = healthy, −10 = 1 SD worse. It also yields a responder definition (ΔGNI ≥ 5). |
| H3 | **Forecaster + counterfactual queries** (footwear, rehab duration) | Lets clinicians ask "what if?". It is honestly labelled as associational, not causal, because the data are observational. |
| H4 | **VAE + GMM phenotypes feeding the forecaster as a mixture-of-experts gate** | Different recovery mechanisms get specialised sub-models. |
| H5 | **Neural-ODE latent dynamics + Bayesian optimisation of rehab schedule** | Continuous-time recovery dynamics, plus an optimiser that proposes visit timing. This needs denser longitudinal data than GaitRec provides, so it is future work. |

## 4. Chosen methodology and justification

**H1 + H2 + H3**: GaitTwin forecaster, GNI outcome layer and counterfactual simulator. It is benchmarked against S1, S2, a diagnosis × time mean-change model and the essential **persistence (no-change) baseline**.

| Criterion | Why H1+H2+H3 |
|---|---|
| Interpretability | GNI is a single, clinically scaled score. Waveform forecasts can be read phase by phase. Occlusion maps show which baseline gait phases drive the forecast. |
| Compute | Trains on a laptop CPU in minutes (about 5k pairs). No GPU is needed, which matters for clinical deployment. |
| Data needs | Fits GaitRec's size. A GP (S3) scales poorly to 1,010-dimensional outputs, and the Neural ODE (H5) needs more visits per patient. |
| Expected accuracy | Learns nonlinear, diagnosis-specific recovery and uses the full waveform. Uncertainty is calibrated by conformal prediction. |
| Discovery potential | Counterfactual footwear and time sweeps, and phase-level importance, produce testable biomechanical hypotheses. |

**Evaluation protocol:**
- Patient-level split using GaitRec's own TRAIN/TEST flags. No patient appears in both sets.
- 15 % of training patients are held out for early checks and conformal calibration.
- Primary endpoint: the **last follow-up visit** of each admission.
- Metrics: waveform RMSE (body weight), GNI MAE, R² and Pearson r of the predicted ΔGNI, responder AUROC, and 90 % interval coverage.
- Subject-level paired bootstrap 95 % CIs comparing models.

## 5. Mini abstract

> **Background.** Gait recovery after lower-limb trauma is heterogeneous. Clinicians lack tools that forecast *how* a patient will walk after rehabilitation, so they cannot see it at admission. **Methods.** We used 1,858 patients from the GaitRec database with repeated force-plate assessments during inpatient rehabilitation. We trained GaitTwin, a 1-D convolutional encoder with FiLM conditioning on clinical context, elapsed treatment time and footwear. A five-member heteroscedastic deep ensemble predicts the full 10-channel ground-reaction-force waveform at any follow-up horizon. Forecasts are summarised by a new GRF Normalcy Index (GNI; healthy = 100, 10 points = 1 SD). Intervals are calibrated with split-conformal prediction. We compared it against persistence, mean-change, PCA-Ridge and PCA-gradient-boosting baselines on held-out patients. **Results.** See `results/metrics.json` and the project page. **Conclusion.** Whole-waveform probabilistic forecasting turns one admission gait test into a patient-specific, uncertainty-aware simulation of post-treatment function. This supports goal-setting and early identification of non-responders. Prospective validation and causal designs are needed before clinical use.
