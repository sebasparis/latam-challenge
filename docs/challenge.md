# LATAM Airlines Software Engineer (ML & LLMs) challenge
# Candidate: Sebastián Gallardo París

# Introduction

In this document I will detail the process of resolution of the challenge that I was given, and give justifications on all decisions taken.

The branch history of the corresponding repository should be pretty descriptive and complementary to this discussion.

# Project Structure

TODO: complete


# Step-by-step resolution

## Part 1

### Changelog
- Completed model.py
- Fixed outdated sns syntax in exploration.ipynb barplots.
- Fixed requirements.txt numpy~=1.22.4 -> numpy~=1.23.0 to avoid an inconsistent environment.
- Added xgboost==3.2.0 to requirements.txt
- Updated requirements-test.txt pytest~=6.2.5 -> pytest~=8.0

### Model determination

I evaluated performance by analyzing the F1 scores in the positive (flight delayed) classes, since I see no a priori business reason to prioritize either precision or recall for this model. Both models achieve comparable performance on the available dataset when using balanced classes. I selected XGBoost because flight-delay prediction is a tabular problem where nonlinear relationships and interactions between variables are plausible. XGBoost can model these relationships directly without requiring explicit feature-interaction engineering, and its additional capacity may become useful as additional informative data and features become available.

This comes at the cost of greater model complexity and computational cost, both during training and inference, as well as reduced interpretability relative to logistic regression. In this application, however, the additional inference cost is not operationally significant given the expected prediction volume. If latency or computational efficiency were primary constraints, logistic regression could be a stronger candidate.

**TL;DR: XGBoost with balanced classes was chosen.**

####

## Part 2

### Changelog
- Completed api.py
- Updated fastapi~=0.86.0 -> fastapi~=0.141.0
- Updated pydantic~=1.10.2 -> pydantic~=2.11.0
- Updated uvicorn~=0.15.0 -> uvicorn~=0.35.0
- Added httpx2~=2.12.0

## Part 3

### Changelog

- Created Dockerfile
- Stored image in Google Artifact Registry and deployed using Google Cloud Run.
- Added service URL to stress tests.
- Updated locust~=1.6 -> locust~=2.46.0

## Part 4