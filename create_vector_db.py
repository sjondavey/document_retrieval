import pandas as pd
from pathlib import Path
import numpy as np
# Necessary for the vector database
import uuid
import chromadb
from chromadb.config import Settings
# The embedding models
from sentence_transformers import SentenceTransformer
import openai
from InstructorEmbedding import INSTRUCTOR 
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    '''
    Abstract base class for a wrapper of an embedding model. 
    
    To implement this class, you need to implement the embed method. If your model has the ability to embed a query
    differently to a document, "then you need to implement the embed method for the document and the embed_query method
    for the query.
    
    Each class needs to ensure output of the embedding model, when saved to file, is a COMMA Separated string 
    that can be loaded into a numpy array with the np.fromstring(x[1:-1], sep=', ') command.

    This class creates a separate database for each embedding model. It also stores the embeddings in a csv file 
    in the database folder for reference.
    '''
    def __init__(self, model, database_folder):
        self.model = model
        self.database_folder = Path(database_folder)
        # Create the database folder if it doesn't exist, don't raise the FileExist error if it does
        self.database_folder.mkdir(parents=True, exist_ok=True)
        # create a persistent database
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=database_folder 
        ))
        # These are the column names in the input csv file
        self.id = "Question_ID"
        self.name_faq = "Questions"
        self.name_answer = "Answers"
        # A column with this name is added to the output csv file which contains the input data plus the embeddings
        self.embeddings = "Embeddings"
        # functionally we can choose to embed the data on the Answer or the Question or a concatenation of both. 
        # I have only tested this where the embedding is done on the Answer
        self.name_combined = "Combined"


    def embed(self, text):
        ''' 
        Abstract method that each child needs to implement to ensure the document, once embedded can be saved as a 
        comma separated list to a csv file
        '''
        raise NotImplementedError()

    def embed_query(self, text):
        ''' 
        If the embedding model has the ability to embed a Query differently to a Document, the embed() method needs to 
        work for the document and this method needs to work for the query.
        '''
        return self.embed(text)

    def embed_and_save_knowledge_base(self, text_faq_data_df, embedding_column):
        '''
        Iterates through the input dataframe and creates an embedding using the data in the embedding_column.
        If embedding_column is "Combined", then the embedding is created by concatenating the data in the Answer and
        Question columns. I have only tested this where the embedding is done on the Answer.
        
        The embeddings are saved to a csv file in the database folder along with the database itself.
        '''
        embed_column_to_action_map = {
            self.name_faq: {
                "filepath": self.database_folder / 'data_with_faq_embeddings.csv',
                "collection_name": self.name_faq.lower(),
            },
            self.name_answer: {
                "filepath": self.database_folder / 'data_with_clean_answer_embeddings.csv',
                "collection_name": self.name_answer.lower(),
            },
            self.name_combined: {
                "filepath": self.database_folder / 'data_with_combined_embeddings.csv',
                "collection_name": self.name_combined.lower(),
            },
        }
        
        if embedding_column not in embed_column_to_action_map:
            raise ValueError(f'embedding name must be one of {", ".join(embed_column_to_action_map.keys())}')

        embedding_action = embed_column_to_action_map[embedding_column]
        filepath = embedding_action["filepath"]
        collection_name = embedding_action["collection_name"]

        if embedding_column == self.name_combined:
            text_faq_data_df[self.embeddings] = text_faq_data_df.apply(lambda row: self.embed(row[self.name_faq] + ". " + row[self.name_answer]), axis=1)
        else:
            text_faq_data_df[self.embeddings] = text_faq_data_df[embedding_column].apply(self.embed)

        text_faq_data_df.to_csv(filepath, index=False)
        collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        # I started using the QuestionID as for the database ids but I needed something else when I started chunking the data
        # so I settled on a "random number" for the id. For the analysis stage of this exercise it does not matter but
        # if we were to use this in production, we would need to ensure that the QuestionIDs were included in the metadata.
        for _, row in text_faq_data_df.iterrows():
            collection.add(
                documents=[row[self.name_answer]],
                embeddings=[row[self.embeddings]],
                metadatas=[{'FAQ': row[self.name_faq]}],
                ids=[str(uuid.uuid1())]
            )
        self.client.persist()

        # if embedding_column not in (self.name_faq, self.name_answer, self.name_combined):
        #     raise Exception('embedding name must be one of ' + self.name_faq + ", " + self.name_answer + ", " + self.name_combined)
        # # Embed the data
        # if embedding_column == self.name_faq:
        #     text_faq_data_df[self.embeddings] = text_faq_data_df[self.name_faq].apply(lambda x: self.embed(x))
        #     text_faq_data_df.to_csv(self.database_folder + '/data_with_faq_embeddings.csv', index=False)
        #     collection = self.client.create_collection(name=self.name_faq.lower(), metadata={"hnsw:space": "cosine"})
        # elif embedding_column == self.name_answer:
        #     text_faq_data_df[self.embeddings] = text_faq_data_df[self.name_answer].apply(lambda x: self.embed(x))
        #     text_faq_data_df.to_csv(self.database_folder + '/data_with_clean_answer_embeddings.csv', index=False)
        #     collection = self.client.create_collection(name=self.name_answer.lower(), metadata={"hnsw:space": "cosine"})
        # elif embedding_column == self.name_combined:
        #     text_faq_data_df[self.embeddings] = text_faq_data_df.apply(lambda row: self.embed(row[self.name_faq] + ". " + row[self.name_answer]), axis=1)
        #     text_faq_data_df.to_csv(self.database_folder + '/data_with_combined_embeddings.csv', index=False)
        #     collection = self.client.create_collection(name=self.name_combined.lower(), metadata={"hnsw:space": "cosine"})
        # else:
        #     raise Exception('embedding name must be one of ' + self.name_faq + ", " + self.name_answer + ", " + self.name_combined)

        # for index, row in text_faq_data_df.iterrows():
        #     collection.add(
        #         documents=[row[self.name_answer]],
        #         embeddings=[row[self.embeddings]],
        #         metadatas=[{'FAQ': row[self.name_faq]}],
        #         #ids=[str(row[self.id])]
        #         ids=[str(uuid.uuid1())] # something unique
        #     )
        # self.client.persist()



    def search_collection_using_embedding(self, collection, embedding, num_docs):
        '''
        Provides the consistent output format for the search results.
        '''
        results = collection.query(query_embeddings=embedding, n_results=num_docs)
        data = {
            'Questions': [metadata['FAQ'] for metadata in results['metadatas'][0]],
            'Answers': results['documents'][0],
            'Scores': results['distances'][0]
        }
        df = pd.DataFrame(data)
        return df

    def search_collection_using_text(self, collection, text, num_docs):
        query_embedding = self.embed_query(text)
        return self.search_collection_using_embedding(collection, query_embedding, num_docs)
    
    def embed_test_questions(self, df_containing_labels, column_name, start_row = 0, end_row = 0):
        df = df_containing_labels
        if end_row <= 0:
            end_row = len(df)
        selected_rows = df.loc[start_row:end_row-1].copy()
        # update the selected rows
        selected_rows[self.embeddings] = selected_rows[column_name].apply(lambda x: self.embed_query(x))
        # write the updated rows back to the original DataFrame
        df.loc[start_row:end_row-1, self.embeddings] = selected_rows[self.embeddings]
        return df
        
            
# child of EmbeddingModel that wraps the https://huggingface.co/hkunlp/instructor-large
class InstructorEmbeddingModel(EmbeddingModel):
    ''' 
    Wrapper for instructor_large and instructor_xl. Hugging Face documentation: https://huggingface.co/hkunlp/instructor-large 
    and https://huggingface.co/hkunlp/instructor-xl
    '''
    def __init__(self, model, database_folder):
        super().__init__(model = model, database_folder = database_folder)
        # Not used yet, but for my reference
        self.separator = " "
        # Not used as the datasets are small so I have not vectorised anything. This is left here for my reference
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('running on GPU')    
        else:
            device = torch.device('cpu')
            print('running on CPU')
        self.model.to(device)

    def embed(self, text):
        text = text.replace("\n", " ")
        # I have used the "prompt engineering" suggested in the Information Retrieval section of the Hugging Face documentation
        # https://huggingface.co/hkunlp/instructor-large
        return self.model.encode([['Represent the Medical document for retrieval: ', text]])[0].tolist()

    def embed_query(self, text):
        text = text.replace("\n", " ")
        # I have used the "prompt engineering" suggested in the Information Retrieval section of the Hugging Face documentation
        # https://huggingface.co/hkunlp/instructor-large
        return self.model.encode([['Represent the Medical question for retrieving supporting documents: ', text]])[0].tolist()


class e5EmbeddingModel(EmbeddingModel):
    ''' 
    Wrapper for e5-base-v2 and e5-large-v2. Hugging Face documentation: https://huggingface.co/intfloat/e5-base 
    and https://huggingface.co/intfloat/e5-large
    '''

    def __init__(self, model, database_folder):
        super().__init__(model = model, database_folder = database_folder)
        # Not used yet, but for my reference
        # self.separator = " "        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.auto_model = AutoModel.from_pretrained(self.model)
        

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _model_embed(self, text_with_prefix):
        # can be used to embed a list in the same way by removing the square brackets in the call to self.tokenizer
        batch_query_dict = self.tokenizer([text_with_prefix], max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.auto_model(**batch_query_dict)
        embeddings = self._average_pool(outputs.last_hidden_state, batch_query_dict['attention_mask'])
        query_embeddings_as_nparray = F.normalize(embeddings, p=2, dim=1).detach().numpy()
        return query_embeddings_as_nparray

    def embed(self, text):
        text = text.replace("\n", " ")
        text_with_prefix = 'passage: ' + text
        return self._model_embed(text_with_prefix)[0].tolist()

    def embed_query(self, text):
        text = text.replace("\n", " ")
        text_with_prefix = 'query: ' + text
        return self._model_embed(text_with_prefix)[0].tolist()
    

class AllMiniLML6v2(EmbeddingModel):
    ''' 
    As well as being small and fast, this is also the base model for ChromaDB (see https://docs.trychroma.com/embeddings)
    '''
    def __init__(self, model, database_folder):
        super().__init__(model = model, database_folder = database_folder)
        # Not used yet, but for my reference
        self.separator = ", "
        # Not used yet but here for my reference
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('running on GPU')    
        else:
            device = torch.device('cpu')
            print('running on CPU')
        self.model.to(device)

    def embed(self, text):
        text = text.replace("\n", " ")
        return self.model.encode(text).tolist()


class OpenAIAda(EmbeddingModel):
    def __init__(self, model, database_folder):
        super().__init__(model = model, database_folder = database_folder)
        # Not used yet, but for my reference
        self.separator = ", "
        self.api_key = os.getenv("OPEN_API_KEY")

    def embed(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=self.model)['data'][0]['embedding']
    