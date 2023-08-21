from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import streamlit as st
import pinecone


def connect_pinecone():
    # connect to pinecone environment
    pinecone.init(
        api_key="eba0e7ab-e2d1-4648-bde2-13b7f8db3415",
        environment="northamerica-northeast1-gcp"  # find next to API key in console
    )


def pinecone_create_index():
    index_name = "abstractive-question-answering"

    # check if the abstractive-question-answering index exists
    if index_name not in pinecone.list_indexes():
        # create the index if it does not exist
        pinecone.create_index(
            index_name,
            dimension=768,
            metric="cosine"
        )

    # connect to abstractive-question-answering index we created
    index = pinecone.Index(index_name)
    return index


def query_pinecone(query, retriever, index, top_k):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc


def format_query(query, context):
    # extract passage_text from Pinecone search result and add the <P> tag
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
    # concatinate all context passages
    context = " ".join(context)
    # contcatinate the query and context passages
    query = f"question: {query} context: {context}"
    return query

def generate_answer(query, tokenizer, generator, device):
    # tokenize the query to get input_ids
    inputs = tokenizer([query], max_length=1024, return_tensors="pt").to(device)
    # use generator to predict output ids
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=50)
    # use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer



def main():
    connect_pinecone()
    index_name = "abstractive-question-answering" # has already been created in pinecone
    index = pinecone_create_index()
    
    user_input = st.text_input("Ask a question:")


    with st.form("my_form"):
        submit_button = st.form_submit_button(label='Get Answer')

    #initialize retriever
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load the retriever model from huggingface model hub
    retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)

    #upsertion of index has been done
    #initilaize generator
    # load bart tokenizer and model from huggingface
    tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
    generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)


    if submit_button:
        result = query_pinecone(user_input, retriever, index, top_k=1)
        query = format_query(user_input, result["matches"])
        print(query)
        ans = generate_answer(query, tokenizer, generator, device)
        st.write(ans)





if __name__ == '__main__':
    main()