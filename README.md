# RAG Demo using Watsonx LLMs

## Instructions to set it up on the local environment

* Clone the Git repo and navigate to the content generation folder in the terminal.
* Run the command streamlit run /.../RAG-Demo.py
* You can access the application at http://localhost:8501/
* Update the credentials API Key & URL from BAM/Workbench in the UI and you are all set to start prompting.
* All the hyperparameters are configurable. Play around with different combinations to get the desired results.
* Choose google/flan-t5-xxl as the Watsonx LLM model to get started as it gives good results.
* This setup uses Vectordb to store the embeddings and the Hugging Face model to create embeddings.
