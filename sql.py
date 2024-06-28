import mysql.connector
import openai
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
import os
import re
import streamlit as st
import json
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import functools

# Hardcoded database credentials
DB_HOST = 'HOST'
DB_USER = 'USER'
DB_PASSWORD = 'PASSWORD'

@functools.lru_cache(maxsize=1)

# Function to get database metadata
def get_db_metadata(host, user, password):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        cursor.execute("SHOW DATABASES")
        schemas = cursor.fetchall()

        metadata = {}

        for schema in schemas:
            schema_name = schema[0]
            metadata[schema_name] = {}

            cursor.execute(f"USE {schema_name}")

            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                metadata[schema_name][table_name] = []

                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()

                for column in columns:
                    column_name = column[0]
                    metadata[schema_name][table_name].append(column_name)

        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return {}

    return metadata

# Function to load JSON metadata
def load_json_metadata(file_path):
    try:
        with open('metadata.json', 'r') as file:
            json_metadata = json.load(file)
        return json_metadata
    except Exception as e:
        st.error(f"Error loading JSON metadata: {e}")
        return {}

# Function to merge JSON metadata with dynamic metadata
def merge_metadata(dynamic_metadata, json_metadata):
    for schema, tables in json_metadata.items():
        if schema not in dynamic_metadata:
            dynamic_metadata[schema] = tables
        else:
            for table, columns in tables.items():
                if table not in dynamic_metadata[schema]:
                    dynamic_metadata[schema][table] = columns
                else:
                    dynamic_metadata[schema][table] = list(set(dynamic_metadata[schema][table] + columns))
    return dynamic_metadata

# Summarize metadata to reduce token usage
def summarize_metadata(metadata):
    summary = {}
    for schema, tables in metadata.items():
        summary[schema] = {table: columns for table, columns in list(tables.items())[:5]}  # Limit to 5 tables per schema
    return summary

# Setting up OpenAI GPT-4
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key='YOUR_API_KEY')

prompt_template = """
You are an expert in SQL. Generate an SQL query based on the following input:

Input: {input}

Available Schemas, Tables, and Columns: {metadata}

Make sure to join tables if necessary.

Output:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["input", "metadata"])
chain = LLMChain(llm=llm, prompt=prompt)

# Function to extract SQL query from GPT-4 response
def extract_sql_query(gpt_response):
    code_block_pattern = r'```(?:sql)?\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, gpt_response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return gpt_response.strip()


# Function to reformat dates in SQL query from yyyy-mm-dd to m/d/yyyy
def reformat_sql_dates(sql_query):
    date_pattern = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')
    def replace_date(match):
        year = match.group(1)
        month = str(int(match.group(2)))
        day = str(int(match.group(3)))
        return f"{month}/{day}/{year}"
    return date_pattern.sub(replace_date, sql_query)

# Function to generate SQL query
def generate_sql_query(user_input, metadata):
    summarized_metadata = summarize_metadata(metadata)
    summarized_metadata_str = json.dumps(summarized_metadata, indent=2)
    gpt_response = chain.run({"input": user_input, "metadata": summarized_metadata_str})
    sql_query = extract_sql_query(gpt_response)
    if 'Date' in sql_query:
        sql_query = reformat_sql_dates(sql_query)
    return sql_query

# Function to execute the SQL query
def execute_sql_query(sql_query, host, user, password):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password
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

# Function to parse SQL query and extract used schema, database, tables, and columns
sql_keywords = {'SUM', 'AS', 'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'JOIN', 'ON', 'LIMIT', 'DESC', 'ASC', 'COUNT', 'AVG', 'MAX', 'MIN'}

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
            if column.upper() not in sql_keywords:  # Exclude SQL functions and keywords
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
        elif len(name_parts) == 1:
            used_elements["tables"].add(name_parts[0])

    return used_elements

def validate_columns(used_elements, metadata):
    valid_columns = set()
    for database in used_elements["databases"]:
        for table in used_elements["tables"]:
            if database in metadata and table in metadata[database]:
                for column in used_elements["columns"]:
                    if column in metadata[database][table]:
                        valid_columns.add(column)
    used_elements["columns"] = valid_columns
    return used_elements


# Function to plot data visualization
def plot_data_visualization(df, user_input, chart_type):
    st.subheader("Data Visualization")
    
    columns = df.columns.tolist()
    x_axis = st.selectbox("Select X-axis", options=columns)
    y_axis = st.selectbox("Select Y-axis", options=columns)
    
    # Create the corresponding chart type using Plotly Express
    if chart_type == "scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter plot of {y_axis} vs {x_axis}')
    elif chart_type == "bar":
        fig = px.bar(df, x=x_axis, y=y_axis, title=f'Bar chart of {y_axis} vs {x_axis}')
    elif chart_type == "pie":
        fig = px.pie(df, values=y_axis, names=x_axis, title=f'Pie chart of {y_axis}')
    elif chart_type == "line":
        fig = px.line(df, x=x_axis, y=y_axis, title=f'Line chart of {y_axis} vs {x_axis}')
    
    st.plotly_chart(fig)

# Streamlit UI
st.title("Database Chatbot")

tabs = st.tabs(["Query & Result", "Data Lineage Graph", "Data Visualization"])

# Inputs for Query & Result tab
with tabs[0]:
    st.subheader("Query & Result")
    user_query = st.text_area("Enter your query")

    if st.button("Submit Query"):
        if user_query:
            dynamic_metadata = get_db_metadata(DB_HOST, DB_USER, DB_PASSWORD)
            json_metadata = load_json_metadata('metadata.json')
            metadata = merge_metadata(dynamic_metadata, json_metadata)

            if metadata:
                summarized_metadata = summarize_metadata(metadata)
                sql_query = generate_sql_query(user_query, summarized_metadata)
                st.session_state.sql_query = sql_query
                st.write(f"Generated SQL Query: {sql_query}")

                result, columns = execute_sql_query(sql_query, DB_HOST, DB_USER, DB_PASSWORD)
                if result:
                    st.session_state.result = result
                    st.session_state.columns = columns
                    st.write("Results:")
                    df = pd.DataFrame(result, columns=columns)
                    st.write(df)

                    used_elements = parse_sql_query(sql_query)
                    used_elements = validate_columns(used_elements, metadata)
                    st.session_state.used_elements = used_elements
                    st.write(f"Used Elements: {used_elements}")

# Data Lineage Graph tab
with tabs[1]:
    st.subheader("Data Lineage Graph")

    if 'sql_query' in st.session_state and 'used_elements' in st.session_state:
        used_elements = st.session_state.used_elements
        G = nx.DiGraph()

        # Add nodes and edges for databases, tables, and columns
        for database in used_elements["databases"]:
            G.add_node(database, label='Database', type='database')
        for table in used_elements["tables"]:
            G.add_node(table, label='Table', type='table')
            for database in used_elements["databases"]:
                G.add_edge(database, table)
            for column in used_elements["columns"]:
                G.add_node(column, label='Column', type='column')
                G.add_edge(table, column)

        # Define the position of nodes in a hierarchical layout
        pos = {}
        pos_y = 0

        for node, data in G.nodes(data=True):
            if data['type'] == 'database':
                pos[node] = (0, pos_y)
                pos_y -= 1

        pos_y -= 1  # Add some space between levels

        pos_x = 0
        table_pos = {}
        for node, data in G.nodes(data=True):
            if data['type'] == 'table':
                pos[node] = (pos_x, pos_y)
                table_pos[node] = pos_x
                pos_x += 1

        pos_y -= 1  # Add some space between levels

        for node, data in G.nodes(data=True):
            if data['type'] == 'column':
                table = list(G.predecessors(node))[0]
                pos[node] = (table_pos[table], pos_y)
                pos_y -= 0.5  # Slightly adjust y-position for each column

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="white", font_size=10, font_weight="bold", arrows=True, edge_color="black", font_color="black")
        
        # Annotate node types
        for node, (x, y) in pos.items():
            plt.text(x, y+0.1, G.nodes[node]['label'], fontsize=8, ha='center')

        plt.title("Database Lineage Flow", fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.info("Submit a query in the 'Query & Result' tab to view the data lineage graph.")

# Data Visualization tab
with tabs[2]:
    st.subheader("Data Visualization")
    st.info("Select a query in the 'Query & Result' tab to view data visualization.")

    if 'result' in st.session_state and 'columns' in st.session_state and 'sql_query' in st.session_state:
        result = st.session_state.result
        columns = st.session_state.columns
        sql_query = st.session_state.sql_query
        df = pd.DataFrame(result, columns=columns)
        
        # Add a dropdown for chart type selection
        chart_type = st.selectbox("Select Chart Type", options=["line", "bar", "pie", "scatter"])
        
        plot_data_visualization(df, user_query, chart_type)
