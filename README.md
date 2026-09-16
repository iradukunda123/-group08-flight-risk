# People Analytics: Who's a Flight Risk?

Predicting and explaining employee attrition with machine learning.

**Group 8 — Final Group Project, MSc Big Data Analytics, Adventist University of Central Africa (AUCA)**
September 2026

---

## 1. What this project is about, in plain words

When an employee resigns, the organisation loses knowledge, pays to recruit and train a replacement, and puts extra pressure on the people who stay. HR teams usually find out someone is unhappy only when the resignation letter arrives.

This project asks a simple question:

> Looking only at information HR already has — age, salary, overtime, satisfaction ratings, how long someone has been in the role — can we tell which employees are more likely to leave, and explain why the model thinks so?

We do three things:

1. **Describe** — explore the data to see which employee characteristics go together with leaving.
2. **Segment** — group employees into "personas" without using the leave/stay outcome at all, then check which persona leaves more.
3. **Predict and explain** — train models that estimate how likely each employee is to leave, then open the black box so a human can see the reasoning behind each prediction.

The result is a decision-support tool, **not a verdict**. See Section 8.

---

## 2. The data

| | |
|---|---|
| **Dataset** | IBM HR Analytics Employee Attrition & Performance (public, on Kaggle) |
| **Rows** | 1,470 employees |
| **Columns** | 35 original variables |
| **Target** | `Attrition` — did the employee leave? (Yes / No) |

The data is synthetic: IBM created it to look like a realistic HR dataset. It is not a real company's records, which is what makes it safe to publish and study.

The split we are trying to predict:

| Outcome | Employees | Share |
|---|---|---|
| Stayed | 1,233 | 83.9% |
| Left | 237 | 16.1% |

Only about 1 in 6 employees left. That imbalance shapes everything that follows — a lazy model that predicts "nobody ever leaves" would already be 83.9% accurate and completely useless. This is why we do not judge models by accuracy.

---

## 3. How the data was cleaned

| Check | What we found | What we did |
|---|---|---|
| Missing values | None | Nothing to fix |
| Duplicate rows | None | Nothing to remove |
| Constant columns | `EmployeeCount`, `Over18`, `StandardHours` — same value for every employee | Dropped (they carry zero information) |
| ID column | `EmployeeNumber` is a name tag, not a characteristic | Dropped |
| Outliers | Some very high salaries and long tenures | Kept. A 30-year employee is unusual, not an error. Deleting real people to make charts prettier would bias the results. |
| Text columns | `Department`, `Job Role`, `Marital Status`, etc. | Converted to numbers (one-hot encoding) so algorithms can read them |

After cleaning: **1,470 rows × 31 columns**, becoming **45 columns** once text was encoded.

One important technical detail: all scaling and encoding for the predictive models happens **inside a pipeline fitted on the training data only**. This prevents data leakage — the model accidentally peeking at the answers it is being tested on.

---

## 4. What exploring the data showed

No single factor drives attrition on its own — every correlation with `Attrition` is below 0.3. Attrition is a combination of pressures, not one cause.

The strongest signals:

| Characteristic | Direction |
|---|---|
| Works overtime | Leaves more (strongest single signal, +0.246) |
| Single | Leaves more |
| Fewer total working years | Leaves more |
| Lower job level, lower income | Leaves more |
| Younger | Leaves more |

Read these as **associations, not causes**. Overtime and leaving travel together; that does not prove overtime makes people leave.

---

## 5. Employee personas (unsupervised learning)

Here we deliberately hid the leave/stay answer from the algorithm and asked it to group employees using 11 characteristics covering four themes:

- **Engagement** — Job Involvement
- **Satisfaction** — Job, Environment, Relationship Satisfaction, Work-Life Balance
- **Workload** — OverTime
- **Tenure** — Total Working Years, Years at Company, Years in Current Role, Years Since Last Promotion, Years With Current Manager

**Method:** standardise the features → PCA (to find the main dimensions of variation) → K-Means (to form the groups).

PCA showed the data mostly varies along two axes: the first is tenure and experience, the second is overtime and workplace satisfaction. We tested K = 2 to 8 groups; **K = 2 scored best** (silhouette 0.2176).

The two personas the algorithm found:

| | Persona 1: Early-Tenure | Persona 2: Experienced, Long-Tenure |
|---|---|---|
| Employees | 957 (65.1%) | 513 (34.9%) |
| Avg. total working years | 8.6 | 16.4 |
| Avg. years at company | 3.7 | 13.1 |
| Observed attrition rate | 18.81% | 11.11% |

The algorithm was never told who left — yet the early-tenure group turned out to have an attrition rate roughly **7.7 percentage points higher**. That is a genuine finding worth HR's attention.

One caution: the silhouette score of 0.2176 is low. These are broad, overlapping segments, not two sharply separate species of employee.

---

## 6. Predicting who leaves (supervised learning)

**Setup:** 80% training / 20% test, split so both halves keep the same 16% attrition rate. The test set was locked away and never touched during preprocessing, cross-validation or tuning.

**Handling the imbalance:** every model uses `class_weight="balanced"`, which tells the algorithm that missing a leaver is a more costly mistake than a false alarm.

**Tuning:** 5-fold stratified cross-validation with grid search, optimising F1.

Results on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.765 | 0.375 | 0.702 | 0.489 | 0.624 |
| **SVM (selected)** | 0.823 | 0.459 | 0.596 | 0.519 | 0.569 |
| Random Forest | 0.827 | 0.444 | 0.340 | 0.386 | 0.431 |

What the metrics mean, in HR terms:

- **Recall 0.596** — of the employees who actually left, the model flagged about 6 in 10 in advance.
- **Precision 0.459** — of the employees it flagged, a little under half actually left. The rest are false alarms.
- **F1 0.519** — the balance between those two. This was our selection criterion.

**Why this table proves accuracy is a trap:** Random Forest has the highest accuracy (0.827) *and* the worst performance at the actual job — it catches only 34% of leavers. It scores well by mostly agreeing that people stay.

**Final model:** Support Vector Machine, chosen for the highest F1. Logistic Regression has a better PR-AUC (0.624) and catches more leavers (recall 0.702), so it would be the better choice if an organisation decided false alarms are cheap and missed leavers are expensive. That is a business trade-off, not a purely technical one.

---

## 7. Explaining the predictions (SHAP)

A risk score nobody can explain is useless to an HR manager — and unfair to the employee. Since an SVM has no built-in feature importance, we used SHAP, which measures how much each characteristic pushed a prediction up or down.

Globally, across all employees, the most influential features were:

1. OverTime
2. StockOptionLevel
3. Age
4. EnvironmentSatisfaction
5. TotalWorkingYears
6. NumCompaniesWorked
7. DistanceFromHome
8. RelationshipSatisfaction
9. JobInvolvement
10. JobRole

For one individual — **Employee 357**, the highest-risk person in the test set:

| | |
|---|---|
| Predicted probability of leaving | 93.1% |
| Actual outcome | Left |

What pushed the prediction up: works overtime, age 21, lowest environment satisfaction rating, only 3 total working years, travels frequently, Sales Representative, no stock options, monthly income of 2,174.

That is a readable profile — a young, junior, frequently-travelling employee with low satisfaction and no equity stake. A manager can act on that sentence; they cannot act on "0.931".

Crucially: SHAP explains **what the model did**, not what causes attrition. "OverTime had the largest influence on the model" is not the same statement as "overtime causes people to quit."

---

## 8. Limitations and how this should (and shouldn't) be used

1. **Imbalanced data.** Only 16% of employees left. Predicting a minority class stays hard even with class weighting.
2. **The model is wrong a lot.** It misses roughly 4 in 10 leavers and raises false alarms on more than half of the people it flags.
3. **The personas overlap.** A 0.2176 silhouette means broad tendencies, not clean employee types.
4. **Nothing here is causal.** Every result is an association found in one synthetic dataset.
5. **Synthetic data.** Findings should be re-validated on real organisational data before anyone acts on them.

**Use it like this:** as an early-warning signal that prompts a conversation, a check-in, or a look at workload in a particular team.

**Never like this:** to label an individual a "flight risk", to shape promotion, pay or termination decisions, or as a substitute for actually talking to employees. A false positive here is a real person quietly written off by a machine on the strength of a coin-flip-grade prediction.

---

## 9. Repository contents

| File | Description |
|---|---|
| `group08-flight-risk.ipynb` | Full analysis notebook (cleaning → EDA → clustering → models → SHAP) |
| `Group_8_Employee_Attrition_Report.pdf` | Written final report |
| `Employee-Attrition-Clean.csv` | Cleaned, encoded dataset produced by the notebook |
| `charts/` | Figures exported by the notebook |
| `README.md` | This file |

### To run it:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
jupyter notebook group08-flight-risk.ipynb
```

Then run the cells top to bottom. Random seeds are fixed at 42, so the numbers above should reproduce exactly. The SHAP section is the slowest part — `KernelExplainer` is computationally expensive, which is why it runs on a sampled background set.

---

## 10. The team

Group 8, supervised by **Dr. Lema LOGAMOU SEKNEWNA**.

| Member | Role |
|---|---|
| Mpayimana Iradukunda Robert | Data & Cleaning Lead |
| Ishimwe Divin Claudel | Unsupervised Learning Lead |
| Shimwa Uwase Sylvie | Supervised Learning Lead |
| Hirwa Fabrice (Rugera) | Story & Interpretation Lead — SHAP, report, slides |

---

## 11. One-paragraph summary

We analysed 1,470 employee records to understand attrition. Clustering — done without ever showing the algorithm who left — split the workforce into an early-tenure group (65% of staff, 18.8% attrition) and an experienced long-tenure group (35%, 11.1% attrition). Of three classifiers, an SVM performed best on F1 (0.519), catching about 60% of leavers at roughly 46% precision. SHAP showed overtime, stock options, age, environment satisfaction and total working years mattered most to the model's predictions, and produced a readable explanation for each individual employee. The model is useful as an early-warning signal for HR, and unfit to be the sole basis for any decision about a person.
