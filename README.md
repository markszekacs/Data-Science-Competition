# Data-Science-Competition – 1st Place Solution

*Quick links*  
- [Problem statement](docs/task_description.pdf)  
- [Final report](docs/task1.pdf)  
- [Source code](src/)

This repository contains our winning solution to an insurance analytics case competition organized by *BlackRock* and the *Financial Mathematics Student Organization of Corvinus University of Budapest*.

Our team achieved *1st place* in the final round of the competition.

## Team Composition

Teams were allowed to consist of up to *three members*.  
Our team competed with *two members*, yet successfully achieved *1st place* in the final round.

*Team members:*
- Mark Szekacs
- Marcell Hajdú

---

## Competition Context

The competition consisted of *four rounds*, with approximately *50 teams* participating in total.  
After the preliminary rounds, the *top 5 teams* advanced to the *live final*.

In the final round:

- teams were given a previously unseen dataset and problem statement,
- had *2 hours under strict time pressure* to develop a complete analytical solution,
- and were required to *present and defend* their methodology and results immediately afterwards.

The challenge therefore tested not only technical skills, but also *problem formulation, methodological judgment, and business-oriented decision making under time constraints*.

---

## Key Insight: Prediction vs. Inference

A central strength of our solution was the early recognition that the case consisted of *two fundamentally different types of analytical problems*:

1. *Prediction task*  
   - Objective: accurately predict future outcomes.
   - Primary focus: *out-of-sample performance*.
   - Appropriate tools: supervised machine learning models (e.g. logistic regression, tree-based methods).
   - Evaluation criteria: ROC-AUC, F1-score, robustness across models.

2. *Inference task*  
   - Objective: understand and explain *which factors drive outcomes* and how.
   - Primary focus: *interpretability and statistical significance*.
   - Appropriate tools: interpretable statistical models with valid inference.
   - Evaluation criteria: sign, magnitude, and significance of effects.

Recognizing this distinction allowed us to *select different methodologies for each task*, rather than forcing a single model class onto both problems.

---

## Methodological Philosophy

Our modeling choices were guided by a simple principle:

**Use methods that best support business-relevant decisions, not just technical performance.**


- For prediction, we compared multiple machine learning models to ensure robustness and strong predictive accuracy.
- For inference, we deliberately chose an interpretable survival analysis framework, allowing clear conclusions about the timing of contract cancellations.
- Model selection balanced statistical rigor, interpretability, and feasibility under time pressure.

This separation ensured that:
- predictive models could be used for *risk scoring and operational decisions*, while
- inferential models provided *actionable insights* into underlying drivers.

---

## Main Findings (High-Level)

- Contract composition, premium structure, and payment-related characteristics play a significant role in explaining cancellation behavior.
- Certain administrative and payment method features are associated with *faster cancellation*, while broader coverage and add-on components are linked to *longer survival times*.
- The results were consistent across models and statistically significant despite the limited development time.

All findings were communicated in a concise, decision-oriented presentation during the final.

---

## Repository Contents

- docs/task1.pdf – final report submitted for the competition  
- docs/task1.qmd – Quarto source file for the report  
- src/ – clean Python scripts for modeling and analysis  
- data/ – placeholder folder (raw data not included)  
- outputs/ – generated tables and intermediate results  

Raw datasets are not included in this repository, as they were provided as part of the competition and can be reloaded externally.

---

## Reproducibility

The repository is structured so that the full analysis can be reproduced by:
1. placing the original datasets into the data/ folder,
2. installing the required Python dependencies,
3. running the modeling scripts or rendering the Quarto report.

---

## Final Note

This project demonstrates not only technical modeling skills, but also the ability to:
- correctly frame business problems,
- choose appropriate analytical tools,
- and deliver defensible insights under real-world constraints.

These aspects were key factors in achieving *1st place* in the competition.
