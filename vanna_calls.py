import streamlit as st

from vanna.remote import VannaDefault
from vanna.openai import OpenAI_Chat
from vanna.vannadb import VannaDB_VectorStore


@st.cache_resource(ttl=3600)
def setup_vanna():
    VANNA_MODEL = 'flaunt-v1'
    VANNA_API_KEY = st.secrets.get("VANNA_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    BIG_QUERY_PROJECT_ID = 'flaunt-v1'
    # BIG_QUERY_CREDS_FILE_PATH = st.secrets.get("BIG_QUERY_CREDS_FILE_PATH")

    vn = VannaDefault(api_key=VANNA_API_KEY, model=VANNA_MODEL)
    # vn.connect_to_sqlite("https://vanna.ai/Chinook.sqlite")
    vn.connect_to_bigquery(project_id=BIG_QUERY_PROJECT_ID, cred_file_path='bigquery_credentials.json')

    # The information schema query may need some tweaking depending on your database. This is a good starting point.
    df_information_schema = vn.run_sql("SELECT * FROM `flaunt-v1.barefaced2.INFORMATION_SCHEMA.COLUMNS`")

    # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
    plan = vn.get_training_plan_generic(df_information_schema)
    plan

    # If you like the plan, then uncomment this and run it to train
    vn.train(plan=plan)
    return vn

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    print("Generating sample questions...")
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    print("Generating SQL query...")
    vn = setup_vanna()
    return vn.generate_sql(question=question, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    print("Checking for valid SQL cached ...")
    vn = setup_vanna()
    print("Validating SQL, should return")
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    print("Running SQL query cached ...")
    vn = setup_vanna()
    return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    print("Checking if we should generate a chart cached ...")
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    print("Generating Plotly code cached ...")
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    print("Running Plotly code cached ...")
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    print("Generating followup questions cached ...")
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    print("Generating summary cached ...")
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)

@st.cache_data(show_spinner="Getting Training Data ...")
def list_training_data():
    print("Getting Training Data ...")
    vn = setup_vanna()
    td = vn.get_training_data()
    print("td", td)
    return td