import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Set the page configuration
st.set_page_config(page_title="Universal Data Visualization App", layout="wide")

# Title of the app
st.title("Universal Data Visualization Tool")
st.write("Upload any CSV file, select variables and chart type, and visualize your data.")

# Sidebar header
st.sidebar.header('User Options')

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Function to check if a column is numeric
def is_numeric(column_data):
    try:
        pd.to_numeric(column_data)
        return True
    except ValueError:
        return False

# Check if file is uploaded
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("## Preview of the dataset")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

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

    # Lists of columns
    all_columns = df.columns.tolist()

    # Determine data types dynamically
    numeric_columns = [col for col in all_columns if is_numeric(df[col])]
    categorical_columns = [col for col in all_columns if not is_numeric(df[col])]

    # Initialize variables
    x_variable = y_variable = z_variable = hue_variable = None

    # Chart selection
    chart_type = st.sidebar.selectbox(
        'Select chart type',
        ['Scatter Plot', 'Line Plot', 'Bar Plot', 'Histogram', 'Box Plot', 'Heatmap', 'Pairplot', '3D Scatter Plot']
    )

    # Variable selection based on chart type
    if chart_type == 'Heatmap':
        st.sidebar.subheader('Heatmap Options')
        selected_columns = st.sidebar.multiselect('Select variables for Heatmap (Numerical)', numeric_columns)
        # Chart Title Input
        chart_title = st.sidebar.text_input("Enter chart title", "Heatmap")
        if len(selected_columns) < 2:
            st.error("Please select at least two numerical variables for Heatmap.")
            st.stop()

    elif chart_type == 'Pairplot':
        st.sidebar.subheader('Pairplot Options')
        selected_columns = st.sidebar.multiselect('Select variables for Pairplot', all_columns)
        # Chart Title Input
        chart_title = st.sidebar.text_input("Enter chart title", "Pairplot")
        if len(selected_columns) < 2:
            st.error("Please select at least two variables for Pairplot.")
            st.stop()

    elif chart_type == '3D Scatter Plot':
        st.sidebar.subheader('3D Scatter Plot Options')
        if len(numeric_columns) < 3:
            st.error("Your data must have at least three numerical columns for a 3D Scatter Plot.")
            st.stop()
        x_variable = st.sidebar.selectbox('Select X variable (Numerical)', numeric_columns)
        y_variable = st.sidebar.selectbox('Select Y variable (Numerical)', numeric_columns)
        z_variable = st.sidebar.selectbox('Select Z variable (Numerical)', numeric_columns)
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
        if st.sidebar.checkbox('Add a grouping variable'):
            hue_variable = st.sidebar.selectbox('Select grouping variable', all_columns)
        else:
            hue_variable = None

    else:
        st.sidebar.subheader(f'{chart_type} Options')

        # Chart Title Input
        chart_title = st.sidebar.text_input("Enter chart title", chart_type)

        if chart_type in ['Scatter Plot', 'Line Plot']:
            if len(numeric_columns) < 2:
                st.error("Your data must have at least two numerical columns for this plot.")
                st.stop()
            x_variable = st.sidebar.selectbox('Select X variable (Numerical)', numeric_columns)
            y_variable = st.sidebar.selectbox('Select Y variable (Numerical)', numeric_columns)
            # Custom Axis Labels
            x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
            y_label = st.sidebar.text_input("Enter Y-axis label", y_variable)
            # Grouping variable
            if st.sidebar.checkbox('Add a grouping variable'):
                hue_variable = st.sidebar.selectbox('Select grouping variable', all_columns)
            else:
                hue_variable = None

        elif chart_type == 'Bar Plot':
            if not categorical_columns:
                st.error("Your data must have at least one categorical column for the X-axis.")
                st.stop()
            if not numeric_columns:
                st.error("Your data must have at least one numerical column for the Y-axis.")
                st.stop()
            x_variable = st.sidebar.selectbox('Select X variable (Categorical)', categorical_columns)
            y_variable = st.sidebar.selectbox('Select Y variable (Numerical)', numeric_columns)
            # Custom Axis Labels
            x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
            y_label = st.sidebar.text_input("Enter Y-axis label", y_variable)
            # Grouping variable
            if st.sidebar.checkbox('Add a grouping variable'):
                hue_variable = st.sidebar.selectbox('Select grouping variable', all_columns)
            else:
                hue_variable = None

        elif chart_type == 'Histogram':
            if not numeric_columns:
                st.error("Your data must have at least one numerical column for the Histogram.")
                st.stop()
            x_variable = st.sidebar.selectbox('Select variable for Histogram (Numerical)', numeric_columns)
            # Custom Axis Label
            x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
            y_label = "Frequency"
            # Grouping variable
            if st.sidebar.checkbox('Add a grouping variable'):
                hue_variable = st.sidebar.selectbox('Select grouping variable', all_columns)
            else:
                hue_variable = None

        elif chart_type == 'Box Plot':
            if not categorical_columns:
                st.error("Your data must have at least one categorical column for the X-axis.")
                st.stop()
            if not numeric_columns:
                st.error("Your data must have at least one numerical column for the Y-axis.")
                st.stop()
            x_variable = st.sidebar.selectbox('Select X variable (Categorical)', categorical_columns)
            y_variable = st.sidebar.selectbox('Select Y variable (Numerical)', numeric_columns)
            # Custom Axis Labels
            x_label = st.sidebar.text_input("Enter X-axis label", x_variable)
            y_label = st.sidebar.text_input("Enter Y-axis label", y_variable)
            # Grouping variable
            if st.sidebar.checkbox('Add a grouping variable'):
                hue_variable = st.sidebar.selectbox('Select grouping variable', all_columns)
            else:
                hue_variable = None

    # Plotting
    st.write(f"## {chart_type}")

    try:
        if chart_type == 'Heatmap':
            # Correlation matrix
            corr = df[selected_columns].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title(chart_title)
            st.pyplot(fig)

            # Download buttons
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            buffer_pdf = BytesIO()
            fig.savefig(buffer_pdf, format='pdf', bbox_inches='tight')
            buffer_pdf.seek(0)

        elif chart_type == 'Pairplot':
            # Pairplot
            g = sns.pairplot(df[selected_columns], hue=hue_variable)
            # Set the title
            g.fig.suptitle(chart_title, y=1.02)
            st.pyplot(g)
            # Save the figure
            buffer = BytesIO()
            g.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            buffer_pdf = BytesIO()
            g.savefig(buffer_pdf, format='pdf', bbox_inches='tight')
            buffer_pdf.seek(0)

        elif chart_type == '3D Scatter Plot':
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = pd.to_numeric(df[x_variable])
            y = pd.to_numeric(df[y_variable])
            z = pd.to_numeric(df[z_variable])
            if hue_variable:
                labels = df[hue_variable].unique()
                for label in labels:
                    idx = df[hue_variable] == label
                    ax.scatter(x[idx], y[idx], z[idx], label=str(label))
                ax.legend()
            else:
                ax.scatter(x, y, z)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.set_title(chart_title)
            st.pyplot(fig)

            # Download buttons
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            buffer_pdf = BytesIO()
            fig.savefig(buffer_pdf, format='pdf', bbox_inches='tight')
            buffer_pdf.seek(0)

        else:
            # Other chart types
            fig, ax = plt.subplots()

            if chart_type == 'Scatter Plot':
                sns.scatterplot(data=df, x=x_variable, y=y_variable, hue=hue_variable, ax=ax)

            elif chart_type == 'Line Plot':
                sns.lineplot(data=df, x=x_variable, y=y_variable, hue=hue_variable, ax=ax)

            elif chart_type == 'Bar Plot':
                sns.barplot(data=df, x=x_variable, y=y_variable, hue=hue_variable, ax=ax)

            elif chart_type == 'Histogram':
                sns.histplot(data=df, x=x_variable, hue=hue_variable, kde=True, ax=ax)
                ax.set_ylabel(y_label)

            elif chart_type == 'Box Plot':
                sns.boxplot(data=df, x=x_variable, y=y_variable, hue=hue_variable, ax=ax)

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