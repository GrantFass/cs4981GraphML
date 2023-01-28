---
tags: [Graph ML]
title: 'Graph ML Lab 6: Project Proposal'
created: '2023-01-19T22:09:12.105Z'
modified: '2023-01-19T22:42:01.759Z'
---

# Graph ML Lab 6: Project Proposal
Grant Fass

## Description
Identify a graph machine learning application or area of research that you are interested in for your project. Select an idea that you believe is interesting and potentially impactful.

For you project:
- Identify the problem to be solved. Examples, “apply graph machine learning based recommendation algorithms to patient-record data to identify similar patients”, “ implement a GCN from scratch to experiment with different aggregation and update methods”, “create a knowledge graph from Twitter data.”
- Explain why this problem interests you.
- Identify what methods have been used to solve this or related problems in the past.
- Identify at least one dataset you can use. If your project is more experimental, you can use synthetic data.
- What do you expect to learn from completing this project?
- It’s Ok to leverage existing work, but you need to make a contribution, i.e., create something new, apply an existing method on a distinctly different dataset, or perform exploratory research.


Expecting about 1 page max in bulleted format as described above.

[Pytorch built-in datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)

## Response
- Want to try a problem related to my senior design project. My design group has transcripts from khan academy that are separated by their domain. I would like to try and do something with LDA to find commonalities among the different domains to be able to predict which domains a new article falls into. I would then like to try to predict which domains have overlaps.
- This problem interests me because it is related to work on my senior design project. I think it can help my team make progress in something we do not understand well but is parallel to our project.
- This has been previously somewhat solved by using [normal machine learning with LDA](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28). There is also this [other paper](https://aclanthology.org/2021.findings-acl.230.pdf) that I skimmed the abstract of.
- I can use the dataset we collected of all of the transcripts of videos under khan academy using beautiful soup.
- I expect to get a better idea of how to apply graph machine learning to prediction problems. I also expect to learn more about working with topic modeling and text data.

- id all entities using spacy and set them as nodes. look for subject object pairs over shortest paths. look for tripplets. infer links based on the dependancy parse. Can use djkstras shortest path. Knowladge graph. look at graph query languages. likelyhood of similarity, edge existing, node type. 
