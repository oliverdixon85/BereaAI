import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from templates import hebrew_template, greek_template, apologetics_template, theology_template, therapy_template, history_template, commentary_template


st.title("BereaAI Bible Assistant")

st.write("BereaAI has expertise in Biblical Hebrew, Greek, Apologetics, Theology, Counselling and Church History. \
        Though still in the early stages BereaAI shows promise in delivering nuanced and thorough explanations. \
        The first answer takes a while to load but the consequent answers load much faster. \
        ")

HUGGINGFACEHUB_API_TOKEN = st.secrets('HUGGINGFACEHUB_API_TOKEN')


repo_id = "meta-llama/Llama-2-7b-chat-hf"

st.header("Parameters")
temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
max_new_tokens = st.slider('Max New Tokens', min_value=100, max_value=2000, value=1024, step=50)
top_p = st.slider('Top P', min_value=0.0, max_value=1.0, value=0.95, step=0.05)

llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens, "top_p": top_p}
)

embeddings = HuggingFaceEmbeddings()

prompt_infos = [
    {
        "name": "hebrew",
        "description": "Good for answering questions about Hebrew Old Testament Bible",
        "prompt_template": hebrew_template,
    },
    {
        "name": "greek",
        "description": "Good for answering questions about Greek New Testament Bible",
        "prompt_template": greek_template,
    },
    {
        "name": "apologetics",
        "description": "Good for answering questions directed against the Bible or Christianity",
        "prompt_template": apologetics_template,
    },
    {
        "name": "theology",
        "description": "Good for answering questions about biblical theology",
        "prompt_template": theology_template,
    },
    {
        "name": "therapy",
        "description": "Good for answering questions about mental health or personal issues",
        "prompt_template": therapy_template,
    },
    {
        "name": "history",
        "description": "Good for answering questions about mental health or personal issues",
        "prompt_template": history_template,
    },
    {
        "name": "commentary",
        "description": "Good for answering questions about verses, chapters or books of the Bible",
        "prompt_template": commentary_template,
    },
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")

names_and_descriptions = [
    ("hebrew", ["for questions about hebrew"]),
    ("greek", ["for questions about greek"]),
    ("apologetics", ["for questions directed against the Bible or Christianity"]),
    ("theology", ["for questions about theology"]),
    ("therapy", ["for questions about mental health"]),
    ("history", ["for questions about history"]),
    ("commentary", ["for questions about verses, passages or books of the Bible"]),
]

router_chain = EmbeddingRouterChain.from_names_and_descriptions(
    names_and_descriptions, FAISS, embeddings, routing_keys=["input"]
)


def generate_response(input_text):
    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,)
    st.info(chain.run(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "Type question here")
    submitted = st.form_submit_button("Submit")
    if not text:
        st.info("You forgot to type something")
    elif submitted:
        generate_response(text)

st.markdown("## Examples")
example1 = "Give me a Hebrew word study of Psalm 23"
example2 = "Give me a Greek word study on John 17:1-5"
example3 = "What is the evidence Jesus actually rose from the dead?"
example4 = "I'm feeling really overwhelmed and overcome by anxiety and I don't know what to do"
example5 = "How and when was the canon of the Bible put together?"
example6 = "Explain the Trinity"
example7 = "Give me a commentary on Matthew 5:3-12"

if st.button(example1):
    user_input = example1
if st.button(example2):
    user_input = example2
if st.button(example3):
    user_input = example3
if st.button(example4):
    user_input = example4
if st.button(example5):
    user_input = example5
if st.button(example6):
    user_input = example6
if st.button(example7):
    user_input = example7

