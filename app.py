import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import chardet

# Set the page configuration
st.set_page_config(page_title="Universal Data Visualization App", layout="wide")

# Title of the app
st.title("Universal Data Visualization Tool")
st.write("Upload any CSV file, specify parsing options if needed, select variables of interest, and visualize your data.")

# Sidebar header
st.sidebar.header('User Options')

# Function to detect file encoding
def detect_encoding(uploaded_file):
    raw_data = uploaded_file.getvalue()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    return encoding

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Check if file is uploaded
if uploaded_file is not None:
    # Encoding detection
    detected_encoding = detect_encoding(uploaded_file)
    st.sidebar.write(f"Detected file encoding: **{detected_encoding}**")

    # CSV parsing options
    st.sidebar.subheader("CSV Parsing Options")
    encoding_options = st.sidebar.selectbox("Select file encoding", options=[detected_encoding, 'utf-8', 'ISO-8859-1', 'latin1'])
    delimiter_options = st.sidebar.selectbox("Select delimiter", options=[',', ';', '\t', '|', 'Other'])
    if delimiter_options == 'Other':
        delimiter = st.sidebar.text_input("Enter custom delimiter")
    else:
        delimiter = delimiter_options

    header_option = st.sidebar.radio("Does your file have a header row?", options=['Yes', 'No'])
    if header_option == 'Yes':
        header = 0
    else:
        header = None

    # Read CSV with specified options
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            encoding=encoding_options,
            delimiter=delimiter,
            header=header,
            engine='python',
            on_bad_lines='skip'  # Updated parameter
        )
        st.write("## Preview of the dataset")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

    # Variable selection
    st.sidebar.subheader("Select Variables of Interest")
    all_columns = df.columns.tolist()

    # Multiselect with search functionality for variables
    selected_columns = st.sidebar.multiselect(
        'Select variables to use',
        options=all_columns,
        default=all_columns[:5] if len(all_columns) > 5 else all_columns,
        help="Start typing to search for variables"
    )

    if not selected_columns:
        st.error("Please select at least one variable.")
        st.stop()

    # Update dataframe to include only selected columns
    df = df[selected_columns]

    # Allow users to rename columns
    st.sidebar.subheader("Column Settings")
    rename_columns = st.sidebar.checkbox("Rename columns for clarity")
    if rename_columns:
        new_column_names = {}
        for col in df.columns:
            new_name = st.sidebar.text_input(f"Rename column '{col}'", value=col)
            new_column_names[col] = new_name
        df.rename(columns=new_column_names, inplace=True)
        selected_columns = df.columns.tolist()

    # Handle missing values
    if df.isnull().values.any():
        st.warning("The dataset contains missing values.")
        missing_value_handling = st.sidebar.selectbox(
            "Select how to handle missing values",
            ["Drop rows with missing values", "Fill missing values with zero", "Fill missing values with mean", "Keep missing values"]
        )

        # Handle based on user selection
        if missing_value_handling == "Drop rows with missing values":
            df = df.dropna()
        elif missing_value_handling == "Fill missing values with zero":
            df = df.fillna(0)
        elif missing_value_handling == "Fill missing values with mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif missing_value_handling == "Keep missing values":
            pass
    else:
        st.success("No missing values detected in the dataset.")

    # Function to check if a column is numeric
    def is_numeric(column_data):
        try:
            pd.to_numeric(column_data)
            return True
        except:
            return False

    # Data type correction
    st.sidebar.subheader("Data Type Correction")
    data_type_corrections = {}
    for col in selected_columns:
        current_type = df[col].dtype
        desired_type = st.sidebar.selectbox(f"Select data type for column '{col}'", options=['auto', 'string', 'numeric'], index=0)
        data_type_corrections[col] = desired_type

    for col, dtype in data_type_corrections.items():
        if dtype == 'string':
            df[col] = df[col].astype(str)
        elif dtype == 'numeric':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Update column lists after type correction
    numeric_columns = [col for col in selected_columns if is_numeric(df[col])]
    categorical_columns = [col for col in selected_columns if not is_numeric(df[col])]

    # Initialize variables
    x_variable = y_variable = z_variable = hue_variable = None
    x_label = y_label = z_label = chart_title = ""

    # Chart selection
    chart_type = st.sidebar.selectbox(
        'Select chart type',
        ['Scatter Plot', 'Line Plot', 'Bar Plot', 'Histogram', 'Box Plot', 'Heatmap', 'Pairplot', '3D Scatter Plot']
    )

    # Variable selection based on chart type
    if chart_type == 'Heatmap':
        st.sidebar.subheader('Heatmap Options')
        selected_heatmap_vars = st.sidebar.multiselect('Select variables for Heatmap (Numeric)', numeric_columns)
        if len(selected_heatmap_vars) < 2:
            st.error("Please select at least two numeric variables for Heatmap.")
            st.stop()
        # Chart Title Input
        chart_title = st.sidebar.text_input("Enter chart title", "Heatmap")
        selected_columns_plot = selected_heatmap_vars

    elif chart_type == 'Pairplot':
        st.sidebar.subheader('Pairplot Options')
        selected_pairplot_vars = st.sidebar.multiselect('Select variables for Pairplot', selected_columns)
        if len(selected_pairplot_vars) < 2:
            st.error("Please select at least two variables for Pairplot.")
            st.stop()
        # Grouping variable
        if st.sidebar.checkbox('Add a grouping variable'):
            hue_variable = st.sidebar.selectbox('Select grouping variable', categorical_columns + numeric_columns)
        else:
            hue_variable = None
        # Chart Title Input
        chart_title = st.sidebar.text_input("Enter chart title", "Pairplot")
        selected_columns_plot = selected_pairplot_vars

    elif chart_type == '3D Scatter Plot':
        if len(numeric_columns) < 3:
            st.error("Your data must have at least three numeric columns for a 3D Scatter Plot.")
            st.stop()
        st.sidebar.subheader('3D Scatter Plot Options')
        x_variable = st.sidebar.selectbox('Select X variable (Numeric)', numeric_columns, key='x_var_3d')
        y_variable = st.sidebar.selectbox('Select Y variable (Numeric)', numeric_columns, key='y_var_3d')
        z_variable = st.sidebar.selectbox('Select Z variable (Numeric)', numeric_columns, key='z_var_3d')
        if x_variable == y_variable or x_variable == z_variable or y_variable == z_variable:
            st.error("Please select three different variables for X, Y, and Z axes.")
            st.stop()
        # Custom Axis Labels
        x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
        y_label = st.sidebar.text_input("Enter Y-axis label", y_variable)
        z_label = st.sidebar.text_input("Enter Z-axis label", z_variable)
        # Chart Title Input
        chart_title = st.sidebar.text_input("Enter chart title", "3D Scatter Plot")
        # Grouping variable
        if st.sidebar.checkbox('Add a grouping variable', key='grouping_3d'):
            hue_variable = st.sidebar.selectbox('Select grouping variable', categorical_columns + numeric_columns, key='hue_3d')
        else:
            hue_variable = None
        selected_columns_plot = [x_variable, y_variable, z_variable]

    else:
        st.sidebar.subheader(f'{chart_type} Options')
        # Chart Title Input
        chart_title = st.sidebar.text_input("Enter chart title", chart_type)

        # Variable selection and labels
        if chart_type in ['Scatter Plot', 'Line Plot']:
            if len(numeric_columns) < 2:
                st.error("Your data must have at least two numeric columns for this plot.")
                st.stop()
            x_variable = st.sidebar.selectbox('Select X variable (Numeric)', numeric_columns)
            y_variable = st.sidebar.selectbox('Select Y variable (Numeric)', numeric_columns)
            if not x_variable or not y_variable:
                st.error("Please select both X and Y variables.")
                st.stop()
            # Custom Axis Labels
            x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
            y_label = st.sidebar.text_input("Enter Y-axis label", y_variable)
            # Grouping variable
            if st.sidebar.checkbox('Add a grouping variable'):
                hue_variable = st.sidebar.selectbox('Select grouping variable', categorical_columns + numeric_columns)
            else:
                hue_variable = None
            selected_columns_plot = [x_variable, y_variable]

        elif chart_type == 'Bar Plot':
            if not categorical_columns:
                st.error("Your data must have at least one categorical column for the X-axis.")
                st.stop()
            if not numeric_columns:
                st.error("Your data must have at least one numeric column for the Y-axis.")
                st.stop()
            x_variable = st.sidebar.selectbox('Select X variable (Categorical)', categorical_columns)
            y_variable = st.sidebar.selectbox('Select Y variable (Numeric)', numeric_columns)
            if not x_variable or not y_variable:
                st.error("Please select both X and Y variables.")
                st.stop()
            # Custom Axis Labels
            x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
            y_label = st.sidebar.text_input("Enter Y-axis label", y_variable)
            # Grouping variable
            if st.sidebar.checkbox('Add a grouping variable'):
                hue_variable = st.sidebar.selectbox('Select grouping variable', categorical_columns + numeric_columns)
            else:
                hue_variable = None
            selected_columns_plot = [x_variable, y_variable]

        elif chart_type == 'Histogram':
            if not numeric_columns:
                st.error("Your data must have at least one numeric column for the Histogram.")
                st.stop()
            x_variable = st.sidebar.selectbox('Select variable for Histogram (Numeric)', numeric_columns)
            if not x_variable:
                st.error("Please select a variable for the Histogram.")
                st.stop()
            # Custom Axis Label
            x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
            y_label = "Frequency"
            # Grouping variable
            if st.sidebar.checkbox('Add a grouping variable'):
                hue_variable = st.sidebar.selectbox('Select grouping variable', categorical_columns + numeric_columns)
            else:
                hue_variable = None
            selected_columns_plot = [x_variable]

        elif chart_type == 'Box Plot':
            if not categorical_columns:
                st.error("Your data must have at least one categorical column for the X-axis.")
                st.stop()
            if not numeric_columns:
                st.error("Your data must have at least one numeric column for the Y-axis.")
                st.stop()
            x_variable = st.sidebar.selectbox('Select X variable (Categorical)', categorical_columns)
            y_variable = st.sidebar.selectbox('Select Y variable (Numeric)', numeric_columns)
            if not x_variable or not y_variable:
                st.error("Please select both X and Y variables.")
                st.stop()
            # Custom Axis Labels
            x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
            y_label = st.sidebar.text_input("Enter Y-axis label", y_variable)
            # Grouping variable
            if st.sidebar.checkbox('Add a grouping variable'):
                hue_variable = st.sidebar.selectbox('Select grouping variable', categorical_columns + numeric_columns)
            else:
                hue_variable = None
            selected_columns_plot = [x_variable, y_variable]

    # Plotting
    st.write(f"## {chart_title}")

    try:
        if chart_type == 'Heatmap':
            # Correlation matrix
            corr = df[selected_columns_plot].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title(chart_title)
            st.pyplot(fig)

        elif chart_type == 'Pairplot':
            # Pairplot
            g = sns.pairplot(df[selected_columns_plot], hue=hue_variable)
            # Set the title
            g.fig.suptitle(chart_title, y=1.02)
            st.pyplot(g)

        elif chart_type == '3D Scatter Plot':
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = pd.to_numeric(df[x_variable], errors='coerce')
            y = pd.to_numeric(df[y_variable], errors='coerce')
            z = pd.to_numeric(df[z_variable], errors='coerce')
            valid_idx = x.notnull() & y.notnull() & z.notnull()
            x = x[valid_idx]
            y = y[valid_idx]
            z = z[valid_idx]
            if hue_variable:
                hue_data = df.loc[valid_idx, hue_variable]
                labels = hue_data.unique()
                for label in labels:
                    idx = hue_data == label
                    ax.scatter(x[idx], y[idx], z[idx], label=str(label))
                ax.legend()
            else:
                ax.scatter(x, y, z)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.set_title(chart_title)
            st.pyplot(fig)

        else:
            # Other chart types
            fig, ax = plt.subplots()
            plot_data = df.dropna(subset=selected_columns_plot)
            if chart_type == 'Scatter Plot':
                sns.scatterplot(data=plot_data, x=x_variable, y=y_variable, hue=hue_variable, ax=ax)
            elif chart_type == 'Line Plot':
                sns.lineplot(data=plot_data, x=x_variable, y=y_variable, hue=hue_variable, ax=ax)
            elif chart_type == 'Bar Plot':
                sns.barplot(data=plot_data, x=x_variable, y=y_variable, hue=hue_variable, ax=ax)
            elif chart_type == 'Histogram':
                sns.histplot(data=plot_data, x=x_variable, hue=hue_variable, kde=True, ax=ax)
                ax.set_ylabel(y_label)
            elif chart_type == 'Box Plot':
                sns.boxplot(data=plot_data, x=x_variable, y=y_variable, hue=hue_variable, ax=ax)
            else:
                st.error("Unsupported chart type selected.")
                st.stop()

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(chart_title)
            st.pyplot(fig)

        # Download buttons
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        buffer_pdf = BytesIO()
        fig.savefig(buffer_pdf, format='pdf', bbox_inches='tight')
        buffer_pdf.seek(0)

        # Display download buttons
        st.download_button(
            label="Download image as PNG",
            data=buffer,
            file_name='plot.png',
            mime='image/png'
        )
        st.download_button(
            label="Download image as PDF",
            data=buffer_pdf,
            file_name='plot.pdf',
            mime='application/pdf'
        )

    except Exception as e:
        st.error(f"An error occurred while generating the plot: {e}")

else:
    st.info('Awaiting CSV file to be uploaded.')