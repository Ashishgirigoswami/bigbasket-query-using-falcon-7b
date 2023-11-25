import pandas as pd
from flask import Flask, request, jsonify
from langchain.vectorstores import Qdrant as Qdrant1
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig,pipeline
import transformers
import torch
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
import time
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline
app = Flask(__name__)
#reading file
data = pd.read_csv(r'C:\Users\User\Downloads\bigBasketProducts.csv')

# Combine columns into a meaningful descriptive document
def create_descriptive_document(row):
    # Example: Concatenate 'product_name', 'brand', 'category', 'description' columns
    document = f"Product: {row['product']}\n"
    document += f"Brand: {row['brand']}\n"
    document += f"Category: {row['category']}\n"
    document += f"Description: {row['description']}\n"
    document += f"sale_price: {row['sale_price']}\n"
    document+= f"rating:{row['rating']}\n"
    # Add more columns or modify the concatenation to suit your requirements
    return document

# Apply the function to create descriptive documents for each row
data['combined_description'] = data.apply(create_descriptive_document, axis=1)
doc_list=data['combined_description'].tolist()

dl1=[]
for i in range(0,len(doc_list)):
  dl1.append(Document(page_content=doc_list[i], metadata={"source": "local"}))


embeddings1 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



qdrant1 = Qdrant1.from_documents(
    dl1,
    embeddings1,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="my_documents",
)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


model_name = "tiiuae/falcon-7b-instruct"
# If GPU available use it
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    quantization_config= quantization_config
)
# Set to eval mode
model.eval()
pipe = transformers.pipeline(
    "text-generation",
    model=model,
    max_new_tokens=100,
    temperature=0.2,
    tokenizer=tokenizer,
    repetition_penalty=1.5,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    eos_token_id=tokenizer.eos_token_id,

)

hf_pipeline = HuggingFacePipeline(pipeline=pipe)


template = """You are assitent chatbot who gives answer only from context provided to you. Answer the question  and if the answer is not contained within the text below, say "I dont have information about it in my current Knowledge"
{summaries}
Query:{question}
"""

qa = RetrievalQAWithSourcesChain.from_chain_type(llm=hf_pipeline, chain_type="stuff",
        retriever=qdrant1.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={
    "prompt": PromptTemplate(
    template=template,
    input_variables=["summaries","question"],
    ),
    },
        verbose=False,)



def llmquery(query):
    start_time = time.time()
    try:
        # Record the start time
        response = qa({"question": query})
    except ValueError as e:
        response = str(e)
        print(response)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    return response




@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt_text = data['prompt'] if 'prompt' in data else None

    if prompt_text:
        generated_sequence = llmquery(prompt_text)
        return jsonify({'generated_text': generated_sequence['answer']})
    else:
        return jsonify({'error': 'Prompt text is missing.'}), 400
    
    from flask import Flask

if __name__ == '__main__':
    app.run()

    
