# Track A judge review — 14 October 2026

Reviewed against the competition's `Track A.pdf` brief and the running product.

## Corrected in this review

- Replaced the procedural icon-like heart with a detailed, textured, locally served GLB cutaway. Included CC0 license and disclosed the asset's AI-assisted provenance. Orbit controls support rotation, zoom, touch and reset.
- Connected three selectable 3D locator nodes and vessel buttons to live LAD/LCX/RCA probabilities. These are schematic vessel locators on generic anatomy, not inferred lesion positions; anatomical placement still needs expert review.
- Added overall CAD classification at a disclosed 0.5 threshold alongside all four probabilities.
- Added exact additive CAD log-odds contributions relative to the reference profile. Contributions reconstruct the actual CAD prediction; supplied measurements are displayed alongside their influence. This is a model-specific mathematical decomposition, not SHAP, a causal explanation, or a vessel explanation.
- Exposed all 55 predictors through the clinical form and schema-derived categorical choices. Fixed Female/Fmale encoding. Rejected empty requests, unknown fields and invalid categorical/nonfinite numeric values. Clear previous outputs when a measurement changes.
- Corrected validation presentation: the ten-fold experimental winners are not the pipelines loaded by the current API. The product now describes the five-fold serving-pipeline results from `reports/champion_models.json`.
- Removed numbered sections, competition labels and cohort badges from the main journey. Added larger inputs and a robot doctor guide that gets out of the way when a field is focused.

## Remaining limitations by judging category

| Criterion | Evidence / remaining work |
|---|---|
| Prediction (30%) | Four saved pipelines and stored fold metrics exist; target columns are excluded. No external validation, calibrated risk, uncertainty intervals or exact published-benchmark superiority. |
| Visualization (25%) | Detailed GLB, orbit/zoom/reset, selectable probability-coloured nodes. Locator positioning is schematic; AI-assisted cutaway is not medically validated and does not contain separately named coronary artery meshes. |
| Interpretability (20%) | Exact CAD contributions, reference probability and actual feature values are available. Vessel-specific explanations remain a worthwhile extension. |
| Integration (15%) | Local API and browser form are connected. Deployment still requires an accessible API, configured frontend URL/CORS and an operational health check. |
| Implementation (10%) | Source, saved weights, reports, license and documentation exist. Full data validation against clinical units/ranges and external cohort evaluation remain necessary before clinical use. |

## Deliverables

- Working local prototype: present.
- Trained pipelines: present.
- Clinical explanation dashboard: present for CAD.
- Technical report: corrected the abstract's RCA winner to match its results table, documented the actual serving pipelines, anatomy provenance, schematic mapping and exact CAD explanation. Regenerated the PDF and verified its page count remains below six.
- Required **3–10 minute YouTube demonstration**: no completed video or published URL found. See `demo-script.md` for a prepared recording sequence. Publishing requires the user's video/account workflow.

## Verification

Production build, targeted frontend lint and the product rendering check pass. A direct prediction check verifies Female encoding and reconstruction of CAD probability from reference log-odds plus contributions (tolerance 1e-4 for rounded response values). GLB loading, live predictions, colour changes and the robot doctor were visually checked in Chrome. The obsolete starter-skeleton test was replaced with a check that the initial product screen requests input, keeps its disclaimer and does not invent patient predictions. Both pages of the regenerated report were rendered and visually reviewed.
