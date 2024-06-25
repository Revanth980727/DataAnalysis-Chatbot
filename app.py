import os
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st
import json
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px

# Function to extract metadata from CSV and Excel files
@st.cache_data
def extract_metadata(source_folder, limit_columns=5, limit_rows=5):
    metadata = {}
    supported_files = [file for file in os.listdir(source_folder) if file.endswith(('.csv', '.xls', '.xlsx'))]
    
    for file in supported_files:
        file_path = os.path.join(source_folder, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=limit_rows)
            metadata[file] = {"columns": df.columns.tolist()[:limit_columns]}
        else:  # Excel files
            xls = pd.ExcelFile(file_path)
            metadata[file] = {}
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=limit_rows)
                if 'Unnamed' in df.columns[0]:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=limit_rows, header=1)
                metadata[file][sheet_name] = df.columns.tolist()[:limit_columns]
    
    return metadata

# Function to create Pandas DataFrame agent from a CSV or Excel file
def create_pd_agent(source_folder, query, metadata):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key='API_KEY')

    # Construct the prompt template
    template = """
    Query: {query}

    Metadata:
    {metadata}

    Based on the query and metadata, which file is most relevant to answer the query?
    If it's an Excel file, also specify which sheet(s) might be relevant.
    Respond in the following format:
    File: [filename]
    Sheets: [sheet1, sheet2, ...] (if applicable)
    """

    # Format metadata into a string
    metadata_str = ""
    for file, file_data in metadata.items():
        metadata_str += f"File: {file}\n"
        if isinstance(file_data, dict):
            for sheet, columns in file_data.items():
                metadata_str += f"  Sheet: {sheet}\n  Columns: {', '.join(columns)}\n"
        else:
            metadata_str += f"Columns: {', '.join(file_data['columns'])}\n"
        metadata_str += "\n"

    # Create a prompt with the user query and metadata
    prompt = PromptTemplate(input_variables=["query", "metadata"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Invoke the chain
    response = chain.invoke({"query": query, "metadata": metadata_str})
#    st.write(f"GPT Response: {response}")  # Debug: Display GPT response
    
    # Parse the response
    response_lines = response['text'].strip().split('\n')
    chosen_file = response_lines[0].split(': ')[1].strip()
    chosen_sheets = response_lines[1].split(': ')[1].strip().split(', ') if len(response_lines) > 1 else None

    if not chosen_file or chosen_file not in metadata:
        st.error(f"No suitable file found in '{source_folder}' for the query: {query}")
        return None, None, None

    selected_file = os.path.join(source_folder, chosen_file)

    # Read the selected file into a Pandas DataFrame
    try:
        if chosen_file.endswith('.csv'):
            df = pd.read_csv(selected_file)
        else:  # Excel files
            if chosen_sheets and len(chosen_sheets) == 1:
                df = pd.read_excel(selected_file, sheet_name=chosen_sheets[0])
                if 'Unnamed' in df.columns[0]:
                    df = pd.read_excel(selected_file, sheet_name=chosen_sheets[0], header=1)
            else:
                xls = pd.ExcelFile(selected_file)
                df = pd.read_excel(xls, sheet_name=None)
                if chosen_sheets:
                    df = {sheet: df[sheet] for sheet in chosen_sheets if sheet in df}
                # Combine all sheets into a single DataFrame
                df = pd.concat(df.values(), keys=df.keys()).reset_index(level=0).rename(columns={'level_0': 'Sheet'})
    except Exception as e:
        st.error(f"Error reading file '{chosen_file}': {str(e)}")
        return None, None, None
    
    st.write(f"Using file '{selected_file}'.")
    if chosen_sheets:
        st.write(f"Sheets: {', '.join(chosen_sheets)}")
    
    agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
    return agent, df, chosen_file

# Function to query the agent
def query_pd_agent(agent, query):
    prompt = (
        """
        If the query requires creating a table, reply as follows:
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
        
        If the query is not asking for a table but requires a response, reply as follows:
        {"answer": "answer"}
        
        Example:
        {"answer": "The product with the highest sales is 'Minions'."}
        
        Ensure your response is in valid JSON format.
        Let's think step by step.

        Here is the query: 
        """
        + query
    )
    response = agent.run(prompt)
    return response.__str__()

import re

def decode_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract the answer or table data
        answer_match = re.search(r'{"answer":\s*"(.+?)"}', response)
        if answer_match:
            return {"answer": answer_match.group(1)}
        
        table_match = re.search(r'{"table":\s*({.+})}', response)
        if table_match:
            try:
                return {"table": json.loads(table_match.group(1))}
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return the raw response as an answer
        return {"answer": response}

def write_response(response_dict: dict):
    if "answer" in response_dict:
        st.write(response_dict["answer"])
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
        return df
    return pd.DataFrame()

def plot_data_visualization(df, user_input):
    st.subheader("Data Visualization")
    columns = df.columns.tolist()
    x_axis = st.selectbox("Select X-axis", options=columns)
    y_axis = st.selectbox("Select Y-axis", options=columns)
    
    chart_type = "scatter"
    if "bar chart" in user_input.lower():
        chart_type = "bar"
    elif "pie chart" in user_input.lower():
        chart_type = "pie"
    elif "line chart" in user_input.lower():
        chart_type = "line"
    
    if chart_type == "scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter plot of {y_axis} vs {x_axis}')
    elif chart_type == "bar":
        fig = px.bar(df, x=x_axis, y=y_axis, title=f'Bar chart of {y_axis} vs {x_axis}')
    elif chart_type == "pie":
        fig = px.pie(df, values=y_axis, names=x_axis, title=f'Pie chart of {y_axis}')
    elif chart_type == "line":
        fig = px.line(df, x=x_axis, y=y_axis, title=f'Line chart of {y_axis} vs {x_axis}')
    
    st.plotly_chart(fig)

def generate_lineage_graph(file_name, columns):
    G = nx.DiGraph()

    G.add_node(file_name, label='File', type='file')
    for column in columns:
        G.add_node(column, label='Column', type='column')
        G.add_edge(file_name, column)

    # Use a hierarchical layout
    pos = nx.spring_layout(G, k=0.9, iterations=50)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color="lightblue", 
            font_size=10, font_weight="bold", arrows=True, edge_color="gray")

    # Add labels with adjusted positions
    for node, (x, y) in pos.items():
        if G.nodes[node]['type'] == 'file':
            plt.text(x, y, node, ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            plt.text(x, y, node, ha='center', va='center', fontsize=8)

    plt.title("CSV Data Lineage Flow", fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit UI
st.title("CSV Data Analysis and Visualization")

# Define the source folder where CSV files are stored (in the GitHub repository)
source_folder = "source"

# Extract metadata from CSV files
metadata = extract_metadata(source_folder)

tabs = st.tabs(["Data Analysis", "Data Visualization", "Data Lineage"])

# Data Analysis tab
with tabs[0]:
    st.subheader("Data Analysis")
    query = st.text_area("Please let me know your query.")

    if st.button("Submit Query", type="primary"):
        if query:
            agent, df, chosen_file = create_pd_agent(source_folder, query, metadata)
            if agent:
                response = query_pd_agent(agent=agent, query=query)
#                st.write("Raw response:", response)  # Debug: Print raw response
                decoded_response = decode_response(response)
                result_df = write_response(decoded_response)

                # Store session state for visualization and lineage tabs
                st.session_state.result_df = result_df
                st.session_state.used_elements = {"file": chosen_file, "columns": df.columns.tolist()}

# Data Visualization tab
with tabs[1]:
    st.subheader("Data Visualization")
    st.info("Select a query in the 'Data Analysis' tab to view data visualization.")

    if 'result_df' in st.session_state:
        result_df = st.session_state.result_df
        if isinstance(result_df, dict):  # Multiple sheets
            sheet_name = st.selectbox("Select a sheet", list(result_df.keys()))
            df_to_plot = result_df[sheet_name]
        else:
            df_to_plot = result_df
        
        if not df_to_plot.empty:
            plot_data_visualization(df_to_plot, query)
        else:
            st.write("No data available for visualization. Please run a query first.")
    else:
        st.write("No data available for visualization. Please run a query first.")

# Data Lineage tab
with tabs[2]:
    st.subheader("Data Lineage")
    st.info("Submit a query in the 'Data Analysis' tab to view the data lineage graph.")

    if 'used_elements' in st.session_state:
        used_elements = st.session_state.used_elements
        generate_lineage_graph(used_elements["file"], used_elements["columns"])
    else:
        st.write("No data available for lineage graph. Please run a query first.")
