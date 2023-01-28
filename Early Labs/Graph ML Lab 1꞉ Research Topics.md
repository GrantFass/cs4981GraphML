---
tags: [Graph ML]
title: 'Graph ML Lab 1: Research Topics'
created: '2022-11-30T20:59:08.552Z'
modified: '2022-11-30T22:36:27.377Z'
---

# Graph ML Lab 1: Research Topics

## Instructions
Identify 3 areas (ideas) for research or application development for machine learning on graphs. Try to select ideas that you believe are interesting and that have the potential for being impactful.

For each idea write an abstract (paragraph) that includes the following:
- Identify the problem to be solved.
- Explain why this is an important problem. 
- What methods have been used to solve this problem, if any.
- What is the potential impact of solving this problem
- Hypothesize how you might solve this problem.

Note: Expecting about 1-page with 3 paragraphs. No more than $\frac{1}{2}$ page per abstract. This is just a start. You'll be able to update your ideas over time. 

The [following paper](http://graphkernels.cs.tu-dortmund.de/) provides a review of available graph datasets and applications.

If you're interested, you can watch the [Microsoft video](https://www.microsoft.com/en-us/research/video/how-to-write-a-great-research-paper-4/) from Simon Peyton Jones on how to write a great research paper.

Add a twist to existing simple problems, modify them for other domains. If more complex then you could just possibly implement something existing.

## Submission
- Submit a PDF of the ipynb notebook file after running through it.
- Submit the three abstracts.

## Abstract 1: Performing Text Summarization with Graphs
- Inspiration from the stanford article [*Learning Sub-structures of Document Semantic Graphs for Document Summarization*](https://www-cs-faculty.stanford.edu/people/jure/pubs/nlpspo-linkkdd04.pdf)
- The main problem that this paper is investigating is Text Summarization.
- Text summarization is an important topic to a variety of industries. It is one of those topics that is garnering a lot of traction in a wide variety of audiences. There are two main approaches, abstractive and extractive text summarization. Abstractive methods are usually more complex but also result in better summaries when compared to their extractive counterparts.
- Some of the common methods of text summarization are LexRank, PageRank, and TextRank methods on the extractive side. Extractive models include Transformers, Latent Dirirchlet Allocation based Models, and many others. Transformers are some of the easiest to use and the most performant among these current extractive models.
- This topic peaked my interest as it is highly related to my current senior design project. In my senior design project we are trying to summarize lectures and transcripts as a form of study aid. We are currently doing this utilizing transformers.
- The potential impact of solving the problem of text summarization using graphs is that we would be better able to accuratly map the interactions between sentences in the document. This would aid in creating better summaries that may be even shorter than they are now. For example, current models often have trouble summarizing content that is more mathematical in nature or which includes equations. It also has difficulties if related topics are separated or intermixed with other topics. By representing the document as a graph we may be better able to summarize topics and find what is related. 
- If I had to solve this problem myself I may try something like using Latent Dirirchlet Allocation to first classify the document into a set of topics. I would likely then do another pass with LDA to classifiy each sentence into one of the topics or sets of topics. Then we could set up a graph with each sentence being a node and each edge being how much that sentence relates to another. This may even be able to be set up in a directional format. I am not necessarily sure where to go from there though as I do not understand graph machine learning well enough yet.

## Abstract 2: Tracking Metallurgical Performance with Graphs
- Inspiration from my Data Science Practicum project last term. I worked with a company called Scot Forge. They tasked our group with investigating some strength and performance tests for metals. They had noticed that there was a performance decrease over time. They wanted us to investigate what factors may be impacting this. They then gave us massive amounts of data that included things like elemental makeup, heat treatment times, test results, and more. There was a lot of data that was highly related but we had to reduce down or ignore as we were unsure how to parse it at the time.
- The problem to be solved in this case would be if we could better estimate end performance for some metal by tracking the process it went through as a graph. 
- This is important because many fields rely on having parts that are structurally sound and meet certain specifications. Being able to get more performance out of the same metal would be highly beneficial. Additionally, this may help in determining which metals would not be useful and would end up failing before time was spent working with them. This would therefore save time and money. This has actually been a pain point for many foundries and forges for a long time according to a Metallurgical Engineer at Scot Forge.
- Many methods have been used to try and solve this problem. Our group alone tried to estimate performance using neural networks, random forests, decision trees, and 9 other models. We were limited to estimating based on elemental makeup alone as well as total heat treatment time due to the dataset sizes and our experience. This means we likely neglected other factors such as multiple heat treatments at different temperatures, enviornment temperatures, and more.
- The impact of solving this problem would be huge. It would help end customer industries that rely on forged parts such as the department of defence, and NASA. It would also help save time, money, and resources.
- To solve this problem I think that a directed graph might be able to be used to model the interactions between the different data points. 

## Abstract 3: Estimating House Pricing using Graphs
- The problem is that often times house prices are likely based on geographical location including what sort of attractions are in the area, other nearby houses, other schools, the state, taxes, and then the quality of the house.
- This is an important problem to solve since it would help better estimate house prices and may even help predict which areas will go up in price over time. This is likely of more interest to the financial sector and other investment groups.
- I think that most of the current approaches are using either a proprietary solution or more of a traditional approach using tabular data. We actually looked at a simmilar problem with the sacremento housing dataset in CS 3300 where we had tabular IID data about the houses features and its location. We were then able to reduce the location to just a street or zipcode and use that as a factor for estimation.
- The impact of solving this problem would be that you could better predict housing markets and know what houses to buy to maximize your investment or build more equity.
- To solve this problem I think you could create a graph where each node was a house, school, business, or landmark. You could then use the edges as the distance each of these other nodes is from the node you wanted to put up on the market. This would give you a graph basically containing all of the information about a neighborhood including what schools, businesses, attractions, and houses were nearby. This would likely give a much more accurate model for house price compared to trying to use just zip code or city street.

## Feedback
I liked this lab. It was interesting thinking about what sort of problems might be able to be solved using graphs compared to traditional methods.


















