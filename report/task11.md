 

# Task 1 Sentiment Analysis

## 1. Problem and design

The given data contain a mixture of movie reviews and email style spam. Spam data was given random binary labels, which means the `label` column does not provide useful information for those rows. Ised one supervised classifier for all three groups, it would create a flawed decision boundary. When data is checked, found that every spam row came from corporate emails, each starting with a `Subject:` header on the first line. This structure does not appear in movie reviews. A two stage pipeline was therefore adopted (Figure 1) Stage B is trained only on rows that Stage A flags as real reviews. Spam rows in the test output receive the dummy label `-1`.

![Figure 1](../figures/task1_pipeline.png)

*Figure 1: Two stage pipeline. Spam rows bypass the sentiment classifier and receive the dummy label `-1`. The four Stage B approaches share Stage A, the star marks B4 (BiLSTM), used for the test set submission.*

## 2. Data inspection

Splits are 11,503 rows train, 1,397 rows val, 1,434 row test with a nearly balanced binary `label`. A regex frequency scan over email content patterns showed that **`Subject:` line starts occur with similar frequency in label 0 and label 1** (means 0.45 vs 0.44 per row): the predicted signature of an email population distributed evenly across the two sentiment classes. URL and e-mail address features are essentially zero, indicating Enron style operational correspondence rather than promotional link spam.