import mysql.connector
import openai
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import os
import re
import streamlit as st
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import functools
import importlib.util
from decimal import Decimal

# Hardcoded database credentials
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'DB_PASSWORD'
DB_NAME = 'telecom_data'

# Synonym dictionary
SYNONYM_DICT = {
    "Charter": "Charter Communications",
    "AT&T": "AT&T",
    "T Mobile": "T-Mobile",
    "Broadband": "Broadband Service",
    "Verizon": "Verizon",
    "competitors": ["AT&T", "T-Mobile", "Verizon", "Broadband"],
    "Port in cable": ["competitors", "AT&T", "T-Mobile", "Verizon", "Broadband"],
    "Port in mobile": ["competitors", "AT&T", "T-Mobile", "Verizon", "Broadband"],
    "Account Number": ["customer", "connection", "user"],
    "Agent Key": ["agent", "account"],
    "Agent Name": ["agent", "connection", "customer"],
    "Location": ["Tech Location", "Customer Location", "Agent Work Location", "address", "place"],
    "Order": ["Tech Order", "Tech Order Number", "Order Number", "Consignment"],
    "Reason": ["Call Reason", "Outage Reason", "Problem", "Issue"],
    "Issue": ["Tech Issue", "Issue Number", "Fault", "Call Reason", "Outage Reason", "Problem", "Reason"],
}

# Initialize OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(openai_api_key='YOUR_API_KEY')

# Initialize ChromaDB for storing query history
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

# Function to get database metadata
@functools.lru_cache(maxsize=1)
def get_db_metadata(host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()

        metadata = {database: {}}

        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            metadata[database][table_name] = []

            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()

            for column in columns:
                column_name = column[0]
                metadata[database][table_name].append(column_name)

        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return {}

    return metadata

# Function to load JSON metadata
def load_json_metadata(file_path):
    try:
        with open(file_path, 'r') as file:
            json_metadata = json.load(file)
        return json_metadata
    except Exception as e:
        st.error(f"Error loading JSON metadata: {e}")
        return {}

# Function to merge dynamic and JSON metadata
def merge_metadata(dynamic_metadata, json_metadata):
    if DB_NAME not in dynamic_metadata:
        dynamic_metadata[DB_NAME] = {}
    
    for table, columns in json_metadata.get(DB_NAME, {}).items():
        if table not in dynamic_metadata[DB_NAME]:
            dynamic_metadata[DB_NAME][table] = columns
        else:
            dynamic_metadata[DB_NAME][table] = list(set(dynamic_metadata[DB_NAME][table] + columns))
    
    return dynamic_metadata

# Function to load examples from a file
def load_examples(file_path):
    spec = importlib.util.spec_from_file_location("examples", file_path)
    examples_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples_module)
    
    examples_list = json.loads(examples_module.examples)
    
    examples = {}
    for example in examples_list:
        examples[example["question"].strip()] = example["sql"].strip()
    
    return examples

# Function to store query history in ChromaDB
def store_query_history(user_query, sql_query, result):
    def convert_decimal(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError

    chroma_db.add_documents([Document(
        page_content=json.dumps({"user_query": user_query, "sql_query": sql_query, "result": result}, default=convert_decimal),
        metadata={"type": "query_history"}
    )])

# Function to summarize metadata
def summarize_metadata(metadata):
    summary = {}
    for schema, tables in metadata.items():
        summary[schema] = {table: columns for table, columns in list(tables.items())[:7]}  # Limit to 5 tables per schema
    return summary

# Initialize ChatGPT model
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key='YOUR_API_KEY')

# Prompt template for generating SQL queries
prompt_template = """
You are an expert in SQL. Generate an SQL query based on the following input:

Input: {input}

Available Schemas, Tables, and Columns: {metadata}

Examples:
{examples}

Common Synonyms: {synonyms}

Context of the query: {context}

Make sure to join tables if necessary and enclose column names in backticks (`).

Output:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["input", "metadata", "synonyms", "examples", "context"])

chain = LLMChain(llm=llm, prompt=prompt)

# Function to preprocess user input
def preprocess_user_input(user_input, synonym_dict):
    for synonym, actual in synonym_dict.items():
        if isinstance(actual, list):
            for item in actual:
                user_input = re.sub(rf'\b{synonym}\b', item, user_input, flags=re.IGNORECASE)
        else:
            user_input = re.sub(rf'\b{synonym}\b', actual, user_input, flags=re.IGNORECASE)
    return user_input

# Function to extract SQL query from GPT response
def extract_sql_query(gpt_response):
    code_block_pattern = r'```(?:sql)?\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, gpt_response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return gpt_response.strip()

# Function to reformat SQL dates
def reformat_sql_dates(sql_query):
    date_pattern = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')
    def replace_date(match):
        year = match.group(1)
        month = str(int(match.group(2)))
        day = str(int(match.group(3)))
        return f"{month}/{day}/{year}"
    return date_pattern.sub(replace_date, sql_query)

# Function to format string
def format_string(input_string):
    if input_string.startswith('"') and input_string.endswith('"'):
        return input_string[1:-1]  # Remove first and last character
    else:
        return input_string  # Return the string as is

# Function to classify query intent
def classify_query_intent(query):
    query_embedding = embeddings_model.embed_query(query)
    intents = ["SELECT", "INSERT", "UPDATE", "DELETE"]
    intent_embeddings = [embeddings_model.embed_query(intent) for intent in intents]
    similarities = cosine_similarity([query_embedding], intent_embeddings)[0]
    return intents[np.argmax(similarities)]

# Function to generate SQL query
def generate_sql_query(user_input, metadata, synonym_dict, examples):
    if user_input in examples:
        return examples[user_input], None

    similar_queries = chroma_db.similarity_search(user_input, k=3, filter={"type": "query_history"})
    similar_examples = chroma_db.similarity_search(user_input, k=3, filter={"type": "example"})
    
    preprocessed_input = preprocess_user_input(user_input, synonym_dict)
    summarized_metadata = summarize_metadata(metadata)
    summarized_metadata_str = json.dumps(summarized_metadata, indent=2)
    examples_str = json.dumps(examples, indent=2)

    relevant_elements = find_relevant_schema_elements(preprocessed_input, metadata)
    query_intent = classify_query_intent(preprocessed_input)
    context = f"Query intent: {query_intent}\nRelevant schema elements: {relevant_elements}"

    combined_examples = [example.page_content for example in similar_examples]
    combined_queries = [query.page_content for query in similar_queries]
    
    combined_content = "\n".join(combined_examples + combined_queries)
    context += f"\nCombined Examples and Similar Queries: {combined_content}"

    gpt_response = chain.run({"input": preprocessed_input, "metadata": summarized_metadata_str, "synonyms": json.dumps(synonym_dict), "examples": examples_str, "context": context})
    sql_query = extract_sql_query(gpt_response)
    sql_query = format_string(sql_query)
    
    if 'Date' in sql_query:
        sql_query = reformat_sql_dates(sql_query)
    
    return sql_query, None

# Function to execute SQL query
def execute_sql_query(sql_query, host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        columns = cursor.column_names
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return [], []

    return result, columns

def generate_summary(question, result):
    def convert_decimal(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError
    
    result_str = json.dumps(result, default=convert_decimal)
    summary_prompt_template = """
    Based on the following question and result, provide a concise, general summary without mentioning specific names just the count of the customers:

    Question: {question}

    Result: {result}

    General Summary:
    """

    summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["question", "result"])
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    summary_response = summary_chain.invoke({
        "question": question,
        "result": result_str
    })

    summary = summary_response["text"].strip()
    return summary




sql_keywords = {'SUM', 'AS', 'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'JOIN', 'TIMESTAMPDIFF', 'ON', 'LIMIT', 'DESC', 'ASC', 'COUNT', 'AVG', 'MAX', 'MIN', 'WITH', 'DISTINCT', 'STR_TO_DATE', 'WHEN', 'CROSS', 'GROUP BY', 'CASE'}

# Function to parse SQL query
def parse_sql_query(sql_query):
    used_elements = {
        "schemas": {"mysql"},
        "databases": set(),
        "tables": set(),
        "columns": set()
    }

    select_regex = re.compile(r'SELECT\s+(.*?)\s+FROM', re.IGNORECASE | re.DOTALL)
    from_regex = re.compile(r'FROM\s+(\S+)', re.IGNORECASE)
    join_regex = re.compile(r'JOIN\s+(\S+)', re.IGNORECASE)
    column_regex = re.compile(r'(\w+)(?:\s+AS\s+\w+)?', re.IGNORECASE)

    select_match = select_regex.search(sql_query)
    if select_match:
        columns_str = select_match.group(1)
        columns = column_regex.findall(columns_str)
        for column in columns:
            if column.upper() not in sql_keywords:
                used_elements["columns"].add(column)

    from_match = from_regex.findall(sql_query)
    join_matches = join_regex.findall(sql_query)
    tables = from_match + join_matches
    for table in tables:
        name_parts = table.split('.')
        if len(name_parts) == 3:
            used_elements["schemas"].add(name_parts[0])
            used_elements["databases"].add(name_parts[1])
            used_elements["tables"].add(name_parts[2])
        elif len(name_parts) == 2:
            used_elements["databases"].add(name_parts[0])
            used_elements["tables"].add(name_parts[1])
        else:
            used_elements["tables"].add(name_parts[0])

    return used_elements

# Function to validate columns
def validate_columns(used_elements, metadata):
    valid_columns = set()
    for table, columns in metadata[DB_NAME].items():
        for column in columns:
            if column in used_elements["columns"]:
                valid_columns.add(column)
    used_elements["columns"] = valid_columns
    return used_elements

# Function to find relevant schema elements
def find_relevant_schema_elements(query, metadata):
    query_embedding = embeddings_model.embed_query(query)
    relevant_elements = []
    for schema, tables in metadata.items():
        for table, columns in tables.items():
            table_embedding = embeddings_model.embed_query(table)
            similarity = cosine_similarity([query_embedding], [table_embedding])[0][0]
            if similarity > 0.5:  # You can adjust this threshold
                relevant_elements.append((schema, table, columns))
    return relevant_elements

# Function to plot data visualization
def plot_data_visualization(df, query, chart_type):
    chart_type = chart_type.lower()
    fig = None

    # Convert all columns to string type
    df = df.astype(str)

    if chart_type == "line":
        fig = px.line(df)
    elif chart_type == "bar":
        fig = px.bar(df)
    elif chart_type == "pie":
        fig = px.pie(df, names=df.columns[0], values=df.columns[1])
    elif chart_type == "scatter":
        fig = px.scatter(df)

    if fig:
        st.plotly_chart(fig)
    else:
        st.error("Unable to create the selected chart type with the given data.")

# Streamlit app
st.title("Data Insights Chatbot")

tabs = st.tabs(["Query & Result", "Data Lineage Graph", "Data Visualization"])

with tabs[0]:
    st.subheader("Query & Result")
    user_query = st.text_area("Enter your query")

    if st.button("Submit Query"):
        if user_query:
            dynamic_metadata = get_db_metadata(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
            json_metadata = load_json_metadata('metadata.json')
            metadata = merge_metadata(dynamic_metadata, json_metadata)

            if metadata:
                summarized_metadata = summarize_metadata(metadata)
                examples = load_examples('examples.py')
                sql_query, stored_result = generate_sql_query(user_query, summarized_metadata, SYNONYM_DICT, examples)
                print(sql_query)
                st.session_state.sql_query = sql_query

                if stored_result:
                    result = stored_result
                    columns = list(result[0].keys()) if result else []
                else:
                    result, columns = execute_sql_query(sql_query, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
                
                    # Store the query and result in ChromaDB
                    store_query_history(user_query, sql_query, result)
                if result:
                    st.session_state.result = result
                    st.session_state.columns = columns

                    # Generate and display the summary first
                    summary = generate_summary(user_query, result)
                    st.write("Summary:")
                    st.write(summary)

                    # Display the results
                    st.write("Results:")
                    df = pd.DataFrame(result, columns=columns)
                    st.dataframe(df.style.set_table_styles([
                        {'selector': 'thead th', 'props': [('font-size', '10pt')]},
                        {'selector': 'tbody td', 'props': [('font-size', '8pt')]}
                    ]).set_properties(**{'max-width': '50px', 'font-size': '8pt'}))

                    used_elements = parse_sql_query(sql_query)
                    used_elements = validate_columns(used_elements, metadata)
                    st.session_state.used_elements = used_elements
                    st.markdown(f"""<p style='font-size: 12px;'>Generated SQL Query: {sql_query}</p>""", unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Data Lineage Graph")

    if 'sql_query' in st.session_state and 'used_elements' in st.session_state:
        used_elements = st.session_state.used_elements
        G = nx.DiGraph()

        for database in used_elements["databases"]:
            G.add_node(database, label='Database', type='database')
        for table in used_elements["tables"]:
            G.add_node(table, label='Table', type='table')
            for database in used_elements["databases"]:
                G.add_edge(database, table)
            for column in used_elements["columns"]:
                G.add_node(column, label='Column', type='column')
                G.add_edge(table, column)

        pos = {}
        pos_y = 0

        for node, data in G.nodes(data=True):
            if data['type'] == 'database':
                pos[node] = (0, pos_y)
                pos_y -= 1

        pos_y -= 1

        pos_x = 0
        table_pos = {}
        for node, data in G.nodes(data=True):
            if data['type'] == 'table':
                pos[node] = (pos_x, pos_y)
                table_pos[node] = pos_x
                pos_x += 1

        pos_y -= 1

        for node, data in G.nodes(data=True):
            if data['type'] == 'column':
                table = list(G.predecessors(node))[0]
                pos[node] = (table_pos[table], pos_y)
                pos_y -= 0.5

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="white", font_size=10, font_weight="bold", arrows=True, edge_color="black", font_color="black")

        for node, (x, y) in pos.items():
            plt.text(x, y + 0.1, G.nodes[node]['label'], fontsize=8, ha='center')

        plt.title("Database Lineage Flow", fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.info("Submit a query in the 'Query & Result' tab to view the data lineage graph.")

with tabs[2]:
    st.subheader("Data Visualization")
    st.info("Select a query in the 'Query & Result' tab to view data visualization.")

    if 'result' in st.session_state and 'columns' in st.session_state and 'sql_query' in st.session_state:
        result = st.session_state.result
        columns = st.session_state.columns
        sql_query = st.session_state.sql_query
        df = pd.DataFrame(result, columns=columns)

        chart_type = st.selectbox("Select Chart Type", options=["line", "bar", "pie", "scatter"])

        plot_data_visualization(df, user_query, chart_type)
