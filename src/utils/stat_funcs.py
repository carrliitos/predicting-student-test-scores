import numpy as np

def cohend(sample_group1, sample_group2):
  """
  Compute Cohen's d for the standardized mean difference between two independent groups.

  Cohen's d is defined as:
    d = (mean(sample_group1) - mean(sample_group2)) / s_pooled
  where s_pooled is the pooled standard deviation of the two groups.

  Parameters
  ----------
  sample_group1 : array-like
    Sample values for group 1 (e.g., exam scores for males). Must be numeric.
  sample_group2 : array-like
    Sample values for group 2 (e.g., exam scores for females). Must be numeric.

  Returns
  -------
  float
    Cohen's d effect size. Positive values indicate group 1 has a higher mean than group 2.

  Notes
  -----
  - Assumes the two samples are independent.
  - Uses the unbiased sample variance (ddof=1) for each group and a pooled standard deviation.
  - If either group has fewer than 2 observations, or if the pooled standard deviation is 0,
    the result may be undefined (division by zero).

  Examples
  --------
  >>> male_scores = np.array([80, 75, 90])
  >>> female_scores = np.array([70, 72, 68])
  >>> cohend(male_scores, female_scores)
  2.236...
  """

  # Calculate the size of the samples
  n1, n2 = len(sample_group1), len(sample_group2)

  # Calculate the variance of the samples
  s1, s2 = np.var(sample_group1, ddof=1), np.var(sample_group2, ddof=1)

  # Calculate the pooled standard deviation
  s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

  # Calculate the means of the samples
  u1, u2 = np.mean(sample_group1), np.mean(sample_group2)

  # Calculate the effecti size
  return (u1 - u2) / s_pooled
