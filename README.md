# Graph Machine Learning Final Project

## Repo Description
This is a repository containing code for the final project in the Graph Machine Learning course at MSOE. This course was held in Spring of 2023 under a CS 4981 Topics in Computer Science course code. Parts of the final project, as well as the course in general, were based on the [instructor's GitHub repo](https://github.com/jayurbain/GraphMachineLearning).

## Project Description
The primary goal of the final project is to identify a way to apply graph machine learning to an area that you are interested in. This should help solve some problem. The problem should be interesting to you and should have data that can be used. More about the requirements for the final project can be found under [Lab 6](https://github.com/jayurbain/GraphMachineLearning/blob/main/labs/Lab%206.%20Graph%20ML%20Research%20Project.pdf).

## Abstract:
***Describe the problem that you are solving, why it's an interesting problem, and why it's an important problem. Clearly state your hypothesis, what your contribution is, and how you will measure results. Provide a sentence summary of results.***
I am attempting to train a supervised topic modeling network using graph machine learning. Current topic modeling networks such as LDA are unsupervised. They are used for clustering a pre-determined number of documents. This does not work well when documents fall into many of the same topics or there is a lot of topic overlap. Additionally, these models only return keywords that represent each topic and do not encode contextual information. My goal is to train a graphical network with each node representing a document. This node would contain the encoded representation of the text for the entire document as well as tag keywords that represent the document. This allows for the prediction of keywords by searching for the most similar documents. I hypothesize that this graphical approach will perform better than a non-graphical network. The primary measurement of model performance will be either f1 score or ROC_AUC_SCORE. Unfortunately, I did not get to the graphical representation stage as I pursued the wrong dataset for too long. This was then followed by multiclass classification output issues during training that prevented completion of models.


## Introduction:
***Provide enough background so that a technical user without graph machine learning experience can get a basic idea of what you are doing. Provide a short summary of prior art.***
Natural Language Processing, or NLP for short, is a rapidly growing domain in machine learning. We have talked about it a fair bit in this class, especially transformer models. Transformer models are often used for text summarization. Popular models can be used fairly quickly through the HuggingFace library. These models can then be adapted to focus in on a project through transfer learning. This is not as readily the case for topic modeling. 

Topic modeling is a domain of Natural Language Processing. Its main goal is to attempt to determine what keywords are present, and or compose, a document. This allows for document classification and clustering. Most topic modeling is unsupervised. This means that there is no ground truth labels passed in during model training. One of the most widely used unsupervised topic models is LDA. That stands for Latent Dirichlet Allocation. Using this model involves specifying a number of topics to look for across documents. New documents are then classified into one of these topics based on the words within it. 

LDA sounds very good so far. It is unsupervised, can be trained quickly, and clusters new documents. It also has some downsides though. One of the biggest downsides is that it is a clustering algorithm at its core. This means that it needs to know exactly how many clusters to attempt to predict for. This results in a key disadvantage of the model not working well on topics that it has not seen before during training. Another downside of LDA comes as a result of it using the words in a document and not the sentences. Using only words in a document results in the model looking for keywords and not context. This means that the model would usually classify articles about math and articles about physics to the same topic. One more key downside of this model is that each topic is ONLY represented by keywords. There is no single 'topic word' that represents an entire topic. For example, a topic may be identified by the keywords 'add', 'subtract', 'divide', and 'constant'. This would make you think that this topic should represent math articles, but you would not be able to say for certain.

The primary goal of this project is to address some of these issues with the LDA model. This will be attempted through a few steps. The first step is to first gague the performance of a standard LDA model. The second step is to train a 'normal' machine learning model using supervised data. The third step would be to attempt to improve the performance of this model by transitioning it to a graphical representation.

Problems like this have been partially solved before for more specific domains. For example, there is a paper on graphical topic detection in online news. One of my personal goals with this project was to attempt to figure out a working model without relying on existing research papers. I made this one of my goals because this is primarly for course work, so I wanted to see if I could figure things out on my own. 

## Methods:
***Describe the methods you will use for your experiements, your dataset, and any necessary data preprocessing.***
The introduction section defined a three step plan to attempt to improve upon the performance of a topic modeling algorithm. The first step was defined as training a standard LDA model. This would be done in order to function as a sort of current 'real-world' baseline. The second step was defined as training a 'normal' machine learning model. This step would be done using supervised data. This allows for the evaluation of metrics such as accuracy, precision, and recall. The third step was defined as taking the 'normal' model and transitioning it to a graphical representation. This would hopefully allow for improved performance over the standard model as it would contain extra contextual information.

### Datasets
One of the first things to do when attempting a project like this is to collect and evaluate what data you have access to. For this project I am using a dataset that a member of my senior design team put together. This dataset is made up of five comma separated value documents. Each of these documents represents a different domain of videos that were scraped from Khan Academy using Beautiful Soup 4. Each video has its information logged into a row in the spreadsheet for the proper domain. This means that each row represents a video and contains the features of 'course', 'unit', 'lesson', 'video_title', 'about', and 'transcript'.

The domains we had data for, and how many videos were in that domain, are as follows:
1. Computing: 263 videos
2. Economics: 819 videos
3. Humanities: 1064 videos
4. Math: 3351 videos
5. Science: 2764 videos

Each of these files were loaded into a pandas dataframe and then joined into one large dataset. A column was added to each record during the join process that denoted the source domain for the given video. This resulted in a dataframe with 8261 records.

I also had access to a TED Talks dataset. This dataset has 2461 entries after dropping null rows. It contains 19 features after the metadata and transcript csv files were joined together based on the video url. Some of the important features in this dataset are 'transcript', 'main_speaker', 'tags', and 'related_talks'.

### Training the standard LDA model
LDA models are unsupervised during training. This means that we only need access to transcripts of videos, or other forms of text, in order to train the model. In our case we have two separate datasets that both contain transcripts. Instead of picking a single dataset to train a model for, we will instead train an LDA model on both of them.

The first step to training the model is to clean up and subsequently tokenize the transcripts. This is done through a preprocessing pipeline that was created. This can be found in the `Preprocessing.py` file. This file has a 'clean' method with an optional tokenize argument. This performs operations on the text such as expanding contractions, removing accented characters, removing xml or html tags, removing extra whitespace, removing stop words, and even performing lemmatization.

After getting a list of tokens that represent each transcript we can then train the model. This is done using gensim and corpora. Corpora allows us to create a dictionary of words (our tokens) across all documents. This is then filtered and run through a doc2bow algorithm to create a corpus. This corpus is then passed into the gensim lda model class along with a number of topics to cluster for. This allows us to train a LDA model.

Once the model is trained we want to visualize it. This can be done through using the PyLDAVis library. This allows us to easily create an interactive visualization of the LDA model. We can hover over each of the topics and see what keywords make up that topic. Larger circles represent keywords that were found in more of the documents. Overlapping circles represent keywords that help compose multiple topics.

We can then run a simple 'test' against this model by performing an inference with a given document. This returns a list of predicted topic numbers and proportions. This allows us to select the topic number witht he highest probability. We can then reverse lookup that topic number to get the keywords that correspond to that topic.

This process was performed in the `KhanAcademyDatasetSetup.ipynb` and `TedTalkDatasetSetup.ipynb` notebook files.

### Training a 'normal' model
In order to train a supervised machine learning model we need two things. The first is to convert our string representation into a numerical representation. The second is to have some sort of ground truth or target that we can use to attempt to predict. In the case of the Khan Academy dataset we have an abundance of possible targets. We could set the target to one of the five domains or even one of the ~20 courses. Unfortunatly, this would not really help us produce keywords. It would also present us with an issue in the future where we could not really expand the model to use a graph. This is due to the Khan Academy dataset not really having enough information to encode a graphical representation. 

Instead it would be better to train a model using the Ted Talk dataset. This is primarily due to the 'related_talks' feature which would let us encode the data as a graphical representation. Additionally each video already has a defined set of keywords in the 'tags' feature. The hardest part will be training a model so that we can use these multiple tags as output. Fortunatly there are some resources on this such as a [StackOverflow Post](https://stats.stackexchange.com/questions/467633/what-exactly-is-multi-hot-encoding-and-how-is-it-different-from-one-hot) and a [Scikit Learn Library](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html).

I did try to train a model on the Khan Academy dataset originally. This was done by creating a record for each sentence across all of the documents after limiting the number of documents to $5000$ due to memory constraints. This turned the $5000$ records representing documents into $355,320$ records representing sentences. Each of these sentences was then encoded using the google universal sentence encoder. This encoder was loaded through tensorflow hub. It turns each sentence into a vector of 512 features. At this point I had a label, from the domain, and an embedding for each sentence. I then reduced the dimensionality using PCA and trained a random forest model. This ended up yielding an $85%$ ROC_AUC_SCORE. As I alluded to earlier, I then got stuck because I could not model relationships between each of the sentence records. More about this process can be found in the `node_embedding.ipynb` notebook.

The next step was to go back to the drawing board and attempt to recreate this process using the TedTalk dataset. Beacuse each transcript had a collection of keywords and other information I figured that it would be better off to model this problem using document based encodings instead of sentence based encodings. This would let me model each Ted Talk transcript document as a node during the graphical representation step. By having each talk be a node I could then use the 'related_talks' feature to connect the given nodes together. 

Each of the documents in the Ted Talk dataset can be encoded using a Doc2Vec model. This model is trained using the "text8" dataset from gensim. During the training stage we can specify the size of the resulting vector encoding as a hyperparameter. This allows us to fine tune our encodings later on. We can then use the `infer_vector` method of the model to get the encoding for the tokens that represent each transcript. 

Next we need to encode our target keywords, also known as our tags. This is done using the [Scikit Learn Multi Label Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html). This module functions similarly to a one hot encoder, but for multi-label classification. We first set up the MultiLabelBinarizer using a set of all unique tags. This comes out to 415 unique tag words in the dataset. Next we use the transform function to transform our stored tags into the corresponding tag vector encodings.

At this stage we now have a method of encoding our text and a method of encoding our tags. This means we now have the two features we need to set up a normal model. See this in the `TedTalkModel.ipynb` notebook.

### Training a Graph Model
Unfortunately, I did not reach this stage. This can be attributed to two reasons. The first is that I spent a lot of time using the Khan Academy dataset instead of the Ted Talk dataset. This resulted in needing to redo work once I switched to the more appropriate dataset. The second reason is that when using the TedTalk dataset I got stuck during the model training phase due to trying to predict multiclass outputs.

## Results:
***Document your experimental trials in tabular and graphical format. Provide your best interpretation of results.***

***Khan Academy Simple Model***
| Metric    | Value |
|-----------|-------|
| ROC_AUC   | 0.85  |
| F1        | 0.78  |
| Accuracy  | 0.84  |
| Precision | 0.74  |
| Recall    | 0.84  |

## Conclusion:
***Summarize your hypothesis, experiment and results.***
As shown by the `node_embedding.ipynb` notebook using the Khan Academy dataset, it is possible to create a model that predicts topics based on sentence embeddings. The model that was trained in the afforementioned notebook was very simple. It listed the same label for each sentence in a document. It also only classified as a deterministic one-vs-rest classification. This yielded an $85%$ ROC_AUC_SCORE.

This model could be improved by giving a topic label for each sentence instead of the entire document. It could also be improved by expanding the number of available label topics to classify against. It should then use a probabilistic approach to classification so that it can return multiple labels for each sentence and or document. These changes would help this model function more like the LDA model it was attempting to emulate.

The `TedTalkModel.ipynb` notebook had fewer conclusions. It was primarily constrained due to time and knowledge. What it accomplished was showing that it is possible to encode entire documents into a variablly sized embedding. It also showed that we can have multiple topics (tags) for each document that we try to predict. This notebook did not make any further progress than that due to getting stuck troubleshooting errors.

## References:
***Provide proper attribution to all of your sources***
- [MonkeyLearn Introduction to Topic Modeling](https://monkeylearn.com/blog/introduction-to-topic-modeling/)
- [TowardsDataScience A Beginner's Guide to Latent Dirichlet Allocation](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2)
- [TowardsDataScience Using LDA Topic Models as a Classification Model Input](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28)
- [ACM A partially supervised cross-collection topic model for cross-domain text classification](https://dl.acm.org/doi/abs/10.1145/2505515.2505556?casa_token=QOvCU3JcODYAAAAA:DT6bO0lWauhFjkWaoc-_xywEkMeK17u1xMyk0ZCbMCDeV0IMTp4STnkndrSvldEBa2Ddm6KKvQ)
- [ScienceDirect A graphical decomposition and similarity measurement approach for topic detection from online news](https://www.sciencedirect.com/science/article/pii/S002002552100356X?casa_token=FLD9zivf578AAAAA:pAX26PY_2X7d1R0PhuN_ItgGsf1n_fOZGj34pNvF9nai3nrFa3X0f4sxILV5NJStvfIIyKA)
- [Powerpoint on Graph models and topic modeling](https://viasm.edu.vn/Cms_Data/Contents/Viasm-EN/Media/file/L5-Graphical-Model-and-TopicModeling.pdf)
- [StackOverflow Multi-hot encoding](https://stats.stackexchange.com/questions/467633/what-exactly-is-multi-hot-encoding-and-how-is-it-different-from-one-hot)
- [Scikit Learn Multi Label Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html)
- [GeeksForGeeks Universal Sentence Encoder Python](https://www.geeksforgeeks.org/word-embedding-using-universal-sentence-encoder-in-python/)
- [Medium Using BERT for classifying documents with long texts](https://medium.com/@armandj.olivares/using-bert-for-classifying-documents-with-long-texts-5c3e7b04573d)
- [Albert Au Yeung BERT Tokenization](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/)
- [StackOverflow Which document embedding model for document similarity](https://stackoverflow.com/questions/65027694/which-document-embedding-model-for-document-similarity)
- [Top2Vec Documentation](https://top2vec.readthedocs.io/_/downloads/en/stable/pdf/)
- [Medium Combining Word Embeddings to form Document Embeddings](https://medium.com/analytics-vidhya/combining-word-embeddings-to-form-document-embeddings-9135a66ae0f)
- [TutorialsPoint gensim doc2vec](https://www.tutorialspoint.com/gensim/gensim_doc2vec_model.htm)