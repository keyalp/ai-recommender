Final Project for the UPC Artificial Intelligence with Deep Learning Postgraduate Course, Spring 2023.

* Authors: [Núria Gonzalez](mailto:nugobo1981@gmail.com), [Jaume Betriu](betriutortjaume@gmail.com), [Miquel Albó](mailto:miquel.albo@estudiantat.upc.edu), [~~Marc Morera~~](yuhu@mmoreram.com)
* Team Advisor: [Paula Gomez Duran](paulagomezduran@gmail.com)
* Date: July 2023

## Table of Contents (Provisional)

* [Introduction](#intro)
* [Overview](#overview)
    * [Setup & Usage](#setup_usage)
    * [Architectures](#architecture)
      * [Random](#architecture)
      * [Popularity](#popularity)
      * [Content Based Filtering](#content_filter)
      * [Collaborative Based Filtering](#collab_filter)
      * [Techniques explored](#tech_exp)
    * [Dataset](#dataset)
      * Features(#data_features)
      * Preprocessing criterion(#data_preproc)
      * Splitting(#data_split)
      * Train(#data_train)
      * Test(#data_test)
      * Eval(#data_eval)
* [Models Exploration:: Understanding]
    * [Random model](#ran_model)
    * [Popularity model](#pop_model)
    * [Factorization Machine Model (Collaborative Filter)](#fm_model)
    * [GCN Model (Collaborative Filter)](#gcn_model)
    * [Deeper Model (Collaborative Filter)](#deep_model)
* [Metrics for evaluation]
    * [Hit rate](#hitrate)
    * [MAP@k](#ndgc)
    * [Intralist](#intralist)
    * [Coverage](#coverage)
    * [BPR](#bpr)

* [Results](#results)

* [Bias problems](#bias)
    * [Cold start](#cold_start)
    * [Filter bubbles](#cold_start)
    * [Low quality](#cold_start)
    * [Lack of diversity](#cold_start)
    * [Scalability](#cold_start)
    * [Non-interpretable results](#cold_start)
    * [Data hungry](#cold_start)
    * [Computational expanse](#cold_start)

* [Conclusions](#conclusions)
* [References](#references)

 

# Introduction <a name="intro"></a>
Recommender systems filter information to avoid overload because of the amount of data generated related to users' interests. Retrieval algorithms used in systems such as Google, Altavista, Grouplens or Amazon are some examples of mapping the available content to the preferences of the users based on their observed behavior about an item.

Our project explores the most mature techniques for the development of decision strategies for users in complex information environments. The project's focus is centered on understanding how they work, the advantages and disadvantages of each model, and alternative proposals.

You will find the papers consulted to carry out the project in the links at the end of the document.
Starting point: [1]

# Setup & usage (keywords) <a name="setup_usage" ></a>


# Architectures<a name="architectures_usage" ></a>
The main purpose of a recommender system (RS) is to predict which item will be clicked next by the user. This is not a guarantee of positive feedback; it just counts what the user probably will click.
In order to achieve this task, we can find two major methods: Content-based and Collaborative-based. However, these methods are complex and require a deep dive into the data and the process, so our starting point has been two simple models to get familiar with RS: Random and Popularity.

If you use online streaming services to watch films, such as Prime or Netflix, professional network such as Linkedin, Youtube, Spotify or whatever, you might recognize the following expressions:
- Here are some recommendations ... [1]
- The most viewed today is ... [2]
- People with similar insterests (have seen, are following...) [3]
- Because you have seen ... [4]

These expressions respond to the method which the recommendation are based:
[1] **Random**:
This is one of the simplest methods, usually selected when new user. A list of k items is randomly selected from the whole item dataset.
![](https://hackmd.io/_uploads/H1KdpUnt2.jpg)

[2] **Popularity**:
This is the second simplest method, user thinks that most viewed items tend to be "the better". We want to be part of the group so if many others have seen I have to see it as well. 
![](https://hackmd.io/_uploads/SyJ2_PhK3.jpg)


[3] **Collaborative** Based Filtering (CF)
This model uses the opinion to recommend items identifying other users with similar taste.
This is the most mature technique and also the most common, we will see more in detail the reasons.
![](https://hackmd.io/_uploads/BkT4_v2th.png)

[4] **Content** Based Filtering (CBF)
The content-based model tries to “understand” why a user interacts with an item: because of user features (as age, genre, address, job), might be the item features (duration, theme, violence, actors…) or any other personal information.
This has a high computational cost because to deal with the large amount of features, the content is limited because it depends on the information that user provide, so normally there is overspecialization and sparsity of data.
![](https://hackmd.io/_uploads/BJnmvt6t2.png)


# Techniques explored <a name="tech_exp"></a>
The two broad approaches in recommender systems are Content-Based(CBF) and Collaborative Filtering(CF).
CF approaches have more advantatges over the amount of data required, while CBF items are vectors of features from user and items, CF just use the historial of user-items interactions. The data domain of CF is scalable to any type of context.  For this reason we found more interesting deep dig into the different approaches of CF architectures.
- Matrix Factorization
- Factorization Machines
- Graph Convolutional Networks


CNN solves cold start problem.
RNNs can help build session-based recommendations without user identification information or even predict what users can buy next based on their click history.
Applying an attention mechanism to the recommender system can help filter out uninformative content and choose the most representative items. It provides good interpretability at the same time.
Deep learning-based recommender systems outperform traditional ones due to their capability to process. DL techniques could be tailored for specific tasks non-linear data.

# References
* https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada
* https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db
* https://arxiv.org/abs/1703.04247
* https://arxiv.org/abs/1708.05031
* https://arxiv.org/abs/2103.03587
* https://arxiv.org/pdf/2007.09036.pdf
* https://datascienceub.medium.com/1-3-recommendation-vanilla-pipeline-for-recommender-systems-rs-ab7425b86d9
* https://datascienceub.medium.com/2-3-recommendation-gcn-for-rs-397e98f37050
* https://medium.com/sciforce/deep-learning-based-recommender-systems-b61a5ddd5456



