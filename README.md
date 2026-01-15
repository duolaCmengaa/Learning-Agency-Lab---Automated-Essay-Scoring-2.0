# Kaggle Competition Learning Agency Lab - Automated Essay Scoring 2.0


## Task Description:

This course project aims to design and implement an open-source Automated Essay Scoring (AES) system to assist in evaluating student writing. Essay grading is a core component of educational assessment but is labor-intensive and limits the timeliness of feedback. Automated scoring methods can support instructors while providing students with consistent and rapid feedback on their writing performance.

The project is based on a large-scale, publicly available dataset of student essays aligned with current educational standards and realistic classroom writing tasks. Unlike earlier open-source efforts that relied on small or non-representative datasets, this dataset includes diverse essay samples across geographic and socioeconomic populations, helping mitigate potential algorithmic bias.

Building on the ideas of the Automated Student Assessment Prize (ASAP) competition, the project explores feature engineering and machine learning models for essay scoring, evaluates model performance against human ratings, and analyzes fairness and generalization across prompts. The final goal is to develop a robust and interpretable automated essay scoring model suitable for educational applications.

## Evaluation

Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two outcomes. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go below 0.

The quadratic weighted kappa is calculated as follows. First, an $N \times N$ histogram matrix $O$ is constructed, such that $O_{i,j}$ corresponds to the number of essay\_id $i$ (actual) that received a predicted value $j$. An $N\text{-by-}N$ matrix of weights, $w$, is calculated based on the difference between actual and predicted values:

$w_{i,j} = \frac{(i-j)^2}{(N-1)^2}$

An $N\text{-by-}N$ histogram matrix of expected outcomes, $E$, is calculated assuming that there is no correlation between values. This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that $E$ and $O$ have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as:

$kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}$

## Dataset Description
The competition dataset comprises about 24000 student-written argumentative essays, each scored on a 1-to-6 scale (see the Holistic Scoring Rubric via [this link](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf)), with the goal of predicting an essay’s score from its text. For file and field details: the `train.csv` (training data) includes `essay_id` (unique essay ID), `full_text` (complete essay content), and `score` (1-6 holistic score); `test.csv` (test data) has the same fields as `train.csv` but excludes `score` (the rerun test set has ~8k entries); `sample_submission.csv` (correct submission format) contains `essay_id` and the predicted `score` (1-6 scale).

## Experimental Results



