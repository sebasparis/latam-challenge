# LATAM Airlines Software Engineer (ML & LLMs) challenge
# Candidate: Sebastián Gallardo París

# Introduction

In this document I will detail the process of resolution of the challenge that I was given, and give justifications on all decisions taken.

The branch history of the corresponding repository should be pretty descriptive and complementary to this discussion.

# Project Structure

This project implements an end-to-end machine learning API for flight-delay prediction. The repository contains the model implementation and supporting data-processing code, a FastAPI application exposing the model through HTTP endpoints, automated unit and API tests, and Docker configuration for packaging the application and its runtime dependencies.

The development workflow is managed through Git and GitHub using a Gitflow-style branch structure, with feature branches integrated into develop through pull requests. GitHub Actions provides the CI/CD pipeline: CI automatically installs the project dependencies and executes the test suite, while CD is triggered after successful CI on develop.

For deployment, the API and model are packaged as a Docker image and stored in Google Artifact Registry. The image is then automatically deployed to Google Cloud Run, which runs the containerized FastAPI service and exposes it through a public HTTPS endpoint. Authentication between GitHub Actions and Google Cloud is implemented using Workload Identity Federation, avoiding long-lived GCP credentials in the repository.

The repository follows a Gitflow-style branching structure. The main branch contains code considered ready for final review and delivery, while develop contains the latest integrated and tested development code. Individual changes are developed in separate feature/ branches, such as feature/part1, feature/part2, etc. Once a feature is completed and its CI checks pass, it is integrated into develop through a pull request.

# API URL

https://latam-api-1052423921685.southamerica-east1.run.app

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
- Deployed service as a Docker image in Google Cloud Run.
- Added service URL to stress tests.
- Updated locust~=1.6 -> locust~=2.46.0

## Part 4

- Completed ci.yml
- Completed cd.yml