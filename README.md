# DataAnalysis-Chatbot

FOR app.py:

1. Enter you OPENAI_API_KEY in app.py at line 38.

2. You can add more data files to the source folder

3. Open CMD and run the following in order

    a. git clone https://github.com/Revanth980727/DataAnalysis-Chatbot.git

    b. pip install -r requirements.txt

    c. streamlit run app.py

4. When you clone make sure you are cloning into a empty folder/directory

FOR sql.py:

1. Enter your DB details at lines 15-18

2. Enter you OPENAI_API_KEY in app.py at line 70.

3. Upload your data to mysql by following the below steps:

    a. Create a Database

    b. Create a table by importing .csv file into mysql workbench using 'Table Data Import Wizard'

    c. Power_PlantsDB: Power_Plants_of_New_Jersey.csv; 
        Overburdened_CommunitiesDB: Overburdened_Communities_under_the_New_Jersey_Environmental_Justice_Law_2021.csv
        Telecom_dataDB: Account.csv; Agent.csv; CallDetails.csv; FactTable.csv; Outage.csv; TechOrder.csv

    c. For an .xlsx file, every sheet should be saved as a seperate .csv file

4. Open CMD and run the following in order

    a. git clone https://github.com/Revanth980727/DataAnalysis-Chatbot.git

    b. pip install -r requirements.txt

    c. streamlit run sql.py