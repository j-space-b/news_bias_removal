# import packages
import warnings
warnings.filterwarnings('ignore')


import sys
sys.path.append('../..')
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, JSONLoader, UnstructuredFileLoader, WebBaseLoader
from langchain.chains import LLMChain, SequentialChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import streamlit as st, tiktoken



st.title("POBI: Parsing Political Bias from News Articles")


# Replicating above functionality but using Streamlit for input
st.subheader('Enter search terms for bias analysis:')

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", value="", type="password")
    st.write(
        "Receive an OpenAPI Key here [link](https://platform.openai.com/account/api-keys)")
    serper_api_key = st.text_input("Serper API Key", value="", type="password")
    news_site = st.selectbox("Choose a News Site", ["foxnews.com", "cnn.com", "npr.org", "abcnews.go.com"])
    st.write(
        "Receive a Serper API Key here [link](https://serper.dev/login)")

    personal_political_assessment = st.slider(
        "Personal Political Score (1= Conservative and 10 = Liberal):",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    num_results = 5

search_query = st.text_input("Type", label_visibility="collapsed")
col1, col2 = st.columns(2)

# If the 'Search' button is clicked
if col1.button("Search"):
    # Validate inputs
    if not openai_api_key.strip() or not serper_api_key.strip() or not search_query.strip():
        st.error(f"Please provide the API keys or the missing search terms.")
    else:
        try:
            with st.spinner("Analyzing articles..."):
                # Show the top X relevant news articles from the previous week using Google Serper API
                search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=serper_api_key)
                result_dict = search.results(f"{search_query} site:{news_site}")
                token_limit = 4000
                if not result_dict['news']:
                    st.error(f"No search results for: {search_query}.")
                else:
                    for i, item in zip(range(num_results), result_dict['news']):
                        url = item.get('link','N/A')
                        if url == 'N/A':
                            continue
                        loader = WebBaseLoader(url) # bs4
                        try:
                            article = loader.load()
                            try:
                                # define model
                                llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', max_tokens=1024,
                                                 openai_api_key=openai_api_key)
                                # Chain 1: Summarize any possible logical fallacies in the text of the article

                                template1 = """You are a university political science professor who uses brevity and inherent logic to assess any political bias from language in a news article {article}. Summarize the article in as few words as possible, being extremely clear. Follow the steps below to output an analysis in one brief sentence. Then list For Potential Logical Fallacies, return a short numbered list of logical fallacies as defined by Aristotle ranked in order of impact on society and ability to mislead, explain why it is a logical fallacy briefly. Do not list any more than 3. If no fallacies are present, return a brief note that no major logical fallacies are found.
                                Return output using the labels below, with a new line of text directly under each label.
                                and rate it on a scale of between 1 and 10, with 1 being extremely conservative and 10 being extremely liberal. 
                                After assessing the article, normalize this in relation to a defined personal political assessment with value {personal_political_assessment} for an audience defines their own political beliefs, on a scale of 1 to 10 with 1 being extremely conservative and 10 being extremely liberal, and explain the strongest reason for the score and its scale in a brief sentence 
                                So for example if the user is defined as a 5 then no difference should be inherent in the assessment of the political bias from the news article.  
                                If the user is defined as a 3 then the political bias assessed in the news article should be increased by a factor of 2. 
                                If a user is defined as a 7 then the political bias assessed in the news article should be decreased by a factor of 2. 
                                Do not explain anything in the first person, just assess in one sentence based on the values and return the result and explain the reason for the political bias assessment. \
                                Summary: \
                                Potential Political Bias Assessment on a scale of 1 to 10: \
                                Reason for Political Bias:\
                                Potential Logical Fallacies:\
                                """
                                prompt_template1 = PromptTemplate(input_variables=["article"], template=template1)
                                chain1 = LLMChain(llm=llm, prompt=prompt_template1, output_key='summary')


                                first = (chain1({"article":article, "personal_political_assessment":{personal_political_assessment}}))
                                second = first['article']
                                st.success(f"Summary: \n\nTitle: {item['title']}\nLink: {item['link']}\n\n{first['summary']}")
                              #  st.success(f"Critique: {item['summary']}\n\nLink: {item['link']}")
                            except Exception as e:
                                if "token" in str(e).lower():
                                    num_results += 1
                                else:
                                    raise
                        except Exception as e:
                            st.exception(f"Error fetching {item['link']}, exception: {e}")
        except Exception as e:
            st.exception(f"Exception: {e}")


