# Research Problem Extraction System for the NLPContributionGraph Shared Task 11 at SemEval-2021

**The NLPContributionGraph (NCG) challenge was organized in SemEval 2021 . The general task information is available here https://ncg-task.github.io/**

We prepared a rule-based approach for this task, utilizing some renowned python libraries and a graph-based method to extract a set of relevant keywords (scientific terms), which builds our research problem results after a group of operations. Using KnowledgeGraph, RAKE, TextRank, and YAKE, we create a collection of keywords from the STANZA text. In most cases, the research problem keywords and terms will emerge in the initial parts of an article. Thus, in this step, we choose the first 20 keywords from each extracted set of the methods mentioned above, as this number approximately provides us with a better result. Finally, we compute the similarity of all remnant keywords with the title and select the most similar keyword to the title as the research problem.

In the evaluation phase, using a python script, we collect all research problem keywords of the test dataset and its related STANZA text. Then after performing our solution on the STANZA text, we check if the result exists in the list of test dataset keywords to regard as the label of the checking record. We consider "1" if the keyword is in the test set and "0" if non of the word in the list equals the keyword. This is why our precision measure is "1", and a reason which tells us F1, recall, and precision are not suitable measures for a ranking problem. Nevertheless, as shown in the table below, we compute the evaluation measures using these estimated labels.

### Dataset
- Training Dataset: https://github.com/ncg-task/training-data
- Test Dataset: https://github.com/ncg-task/test-data

### Evaluation

|           |TextRank|Rake+TextRank|Yake+TextRank|KnowledgeGraph+Rake+TextRank|KnowledgeGraph+Yake+Rake+TextRank|
|:----------|:------:|:-----------:|:-----------:|:--------------------------:|:-------------------------------:|
|F1-score   | 0.38   | **0.38**    |     0.32    |             0.26           |              0.23               |
|Recall     | 0.23   | **0.24**    |     0.19    |             0.15           |              0.13               |
|Precision  | 1.0    | 1.0         |      1.0    |              1.0           |               1.0               |
|Accuracy   | 0.23   | **0.24**    |     0.19    |             0.15           |              0.13               |
