# BigBasket Data Processing and Querying
## Overview
This project involves processing the provided BigBasket data, converting it into a document format by aggregating rows for each column. Subsequently, the Sentence Transformer embedding model is used to transform these documents into embeddings. The embeddings are then stored in a Qdrant database. Additionally, the Falcon-7B-Instruct model is utilized for Language Model (LM) queries, and Langchain is employed for queries related specifically to the BigBasket dataset.

## Steps Taken
### Data Preprocessing

The provided BigBasket data was processed by aggregating rows for each column to create documents.

### Embedding Generation

Utilized the Sentence Transformer embedding model to convert the documents into embeddings, facilitating efficient storage and retrieval.
### Database Integration

Employed Qdrant DB to store the generated embeddings, enabling effective querying and retrieval.
### Querying with Falcon-7B-Instruct

Utilized the Falcon-7B-Instruct model for Language Model queries, enabling diverse question answering capabilities.
### BigBasket-Specific Queries with Langchain

Used Langchain specifically for querying the BigBasket dataset, enabling focused and accurate retrieval of information.

### Technologies Used

Sentence Transformer: For document embedding generation.

Qdrant DB: For efficient storage and retrieval of embeddings.

Falcon-7B-Instruct Model: For Language Model queries.

Langchain: For specific queries related to the BigBasket dataset.

## Code section

### Server File:

The Python file (filename.py) runs a server that contains a webhook "/generate". This endpoint accepts JSON input with a prompt and generates a response to the query in the prompt.

## Running the Server
To run the server and utilize the webhook "/generate", follow these steps:

\Install Dependencies:Use the requirements.txt file to install all necessary dependencies for the project.

### Run the Server:
Execute the Python file containing the server code.

#Using the "/generate" Endpoint:
Send a JSON input with the "Prompt" field to the "/generate" endpoint to receive a response related to the query provided.

### Files in the Repository
notebook.ipynb: Contains the detailed workflow and steps followed in the project.
filename.py: Runs the server and contains the "/generate" webhook for query responses.
requirements.txt: Lists all necessary dependencies and packages required to run the code.

## Example:
![IMG_0200](https://github.com/Ashishgirigoswami/bigbasket-query-using-falcon-7b/assets/43043428/db83e2dd-4c0b-4ba4-9099-059de7c5a16c)

![IMG_0199](https://github.com/Ashishgirigoswami/bigbasket-query-using-falcon-7b/assets/43043428/fb865ae3-bb36-4e7a-ac25-23cf62b720a7)

![IMG_0201](https://github.com/Ashishgirigoswami/bigbasket-query-using-falcon-7b/assets/43043428/b1953e9f-1f78-4d73-a82c-7500ac0d74d3)

![IMG_0202](https://github.com/Ashishgirigoswami/bigbasket-query-using-falcon-7b/assets/43043428/eb03cd15-591c-48dd-b94b-39554434aa70)

![IMG_0205](https://github.com/Ashishgirigoswami/bigbasket-query-using-falcon-7b/assets/43043428/bc74b348-23af-433f-9a39-97fcd027b63b)

![IMG_0204](https://github.com/Ashishgirigoswami/bigbasket-query-using-falcon-7b/assets/43043428/4d25f9c3-9c26-4c2a-87f7-0856d803dc9d)

![IMG_0203](https://github.com/Ashishgirigoswami/bigbasket-query-using-falcon-7b/assets/43043428/d7f9bc91-cb0c-4ce3-a19b-99a63943a6f6)

## Future Enhancements

Finetuning the falcon-7b model.

Introducing more advanced NLP models for improved understanding and accuracy in queries.

Creating end-to-end conversational chatbot.
