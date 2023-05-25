# Evaluate Embedding Models for Document Retrieval


This project, along with a writeup containing the background and context on my blog [$\aleph_1$](wwww.aleph-one.co), is a step on the road to augment Large Language Models (LLMs) with New Data i.e., data that is not in the LLM's training set because it is either new or proprietary. 

In this project we create a performance benchmark so we can test various embedding models ability to perform document retrieval on Frequently Asked Question (FAQ) type data. The exercise consists of:
- Use the [kaggle data on mental health](https://www.kaggle.com/datasets/narendrageek/mental-health-faq-for-chatbot), which consists of 98 Questions paired with their Answer
- Use each embedding model to embed each Answer.
- For each of the 98 Questions, use cosine similarity to retrieve the closest Answer in the vector space and check if this is actually the answer from the input data.

Models are therefore scored out of 98. I evaluate the models
- "all-MiniLM-L6-v2", 
- "text-embedding-ada-002", 
- "instructor_large", 
- "instructor_xl", 
- "e5-base-v2", 
- "e5-large-v2"
I chose these using the [Hugging Face Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard) (MTEB) leaderboard because they were all rated very highly. all-MiniLM-L6-v2 is small, quick and good for debugging my code. New models can be included by creating the appropriate class wrapper of the model in the file "create_vector_db.py".


The structure of this project is very simple. The notebooks
- 01_test_e5_models.ipynb
- 02_test_instructOR_embeddings_in_db.ipynb
- 03_test_vector_db.ipynb

Are simple manual test of various aspects of the models. I created them because I needed something simple to make sure I was calling the models correctly and I could build the correct intuition about them. I am leaving them here because I found them valuable.

The main results are in the notebook
- 10_embedding_model_performance_test.ipynb

You run through it once for each model you want to test. The final results show the score of that model for the test. Finally, I wanted to check how chunking the data that went to the embedding model impacted the ability to generate similar vectors. You can see the results in  
- 50_Appendix_Input_Token_Length.ipynb

Which is largely a copy of the main notebook.
