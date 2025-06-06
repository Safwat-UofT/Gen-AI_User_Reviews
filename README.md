# Gen-AI_User_Reviews  
**Replication Package for:**  
**_Understanding the Challenges and Promises of Developing Generative AI Apps: An Empirical Study_**

This repository contains the replication package for our study on user reviews of generative AI mobile applications. It includes all scripts, sample data, and prompts used across the three main stages of our experiment.

---

## 📁 Repository Structure

### `1_Filter_reviews/`  
This folder contains the scripts and labeled samples used to identify informative reviews.

- `filtering_prompt_P1.py`  
  Prompt script used to filter non-informative reviews using OpenAI's `gpt-4o-mini`.

- `Art_and_design_informative_labeled.py`  
- `Photography_informative_labeled.py`  
- `Productivity_informative_labeled.py`  
  > Each of these files contains a statistically representative sample of user reviews from the respective app categories.  
  > Every entry includes a `unique_id`, the `content` of the review, and a `manual_screening` label (`informative` or `non-informative`).

---

### `2_Extract_topics/`  
This folder contains materials used for topic extraction.

- `topic_extraction_P4.py`  
  Python script containing prompt templates for topic extraction (P2, P3, and P4). The script is configured to run prompt P4.

- `large_samples/`  
  Folder containing large, statistically representative review samples from each app category used as input for the topic extraction prompts.

- `top_topics.csv`  
  Output file listing each app category along with the top extracted topics.

---

### `3_Assign_topics/`  
This folder contains the script for topic assignment.

- `assign_topics_P5.py`  
  Prompt script used to assign extracted topics to all informative reviews using the selected top topics per category.

---

## Notes  
- All reviews used in this package are anonymized and sampled in accordance with ethical research practices.  
- Prompts were run using OpenAI’s `gpt-4o-mini` model unless otherwise specified.

For any questions or replication requests, please contact the corresponding author listed in the paper.
