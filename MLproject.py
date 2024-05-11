import statistics
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import messagebox, ttk
from tkinter import filedialog
from collections import Counter
from scipy.stats import chi2_contingency 

filename = ""
df = pd.DataFrame()  

def upload_file():
    global filename, df
    filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.csv")])
    messagebox.showinfo("File Uploaded", f"File {filename} uploaded successfully.")
    df = pd.read_csv(filename)
    column_names = df.columns.tolist()
    column_select.set('')  # clear the current selection
    column_menu['values'] = column_names
    x_var_menu['values'] = column_names
    y_var_menu['values'] = column_names



def box_plot():
    A = column_select.get()  # Changed from entry.get()
    df[A].fillna(0, inplace=True)
    numbers = df[A].to_list()
    fig, ax = plt.subplots()
    ax.boxplot(numbers)
    ax.set_title('Box Plot of Numbers')
    plt.show()

    A = column_select.get()  # Changed from entry.get()
    df[A].fillna(0, inplace=True)
    numbers = df[A].to_list()
    fig, ax = plt.subplots()
    ax.boxplot(numbers)
    ax.set_title('Box Plot of Numbers')
    plt.show()
def median_smoothing(data, num_bins):
    bin_size = len(data) // num_bins

    mean_smoothed = []

    for i in range(0, len(data), bin_size):
        bin_data = data[i:i+bin_size]

        bin_median = sorted(bin_data)[len(bin_data) // 2]

        mean_smoothed.extend([bin_median] * len(bin_data))

    return mean_smoothed
def mean_smoothing(data, num_bins):
    bin_size = len(data) // num_bins

    smoothed_data = []

    for i in range(0, len(data), bin_size):
        current_bin = data[i:i+bin_size]

        bin_mean = sum(current_bin) / len(current_bin)

        smoothed_data.extend([bin_mean] * len(current_bin))

    return smoothed_data
def boundary_smoothing(data, num_bins):
    bin_size = len(data) // num_bins

    smoothed_data = []

    for i in range(0, len(data), bin_size):
        current_bin = data[i:i+bin_size]

        bin_min = min(current_bin)
        bin_max = max(current_bin)

        smoothed_data.extend([bin_min, bin_max])

    return smoothed_data
def min_max_normalization(data):
    minimum = min(data)
    maximum = max(data)
    range_val = maximum - minimum
    normalized_byMinMax = [(x - minimum) / range_val for x in data]
    return normalized_byMinMax
def z_score_normalization(data):
    mean = np.mean(data)
    sd = np.std(data)
    normalized_zScoure = (data - mean) / sd
    return normalized_zScoure
def normalization_by_decimal_scaling(data):
    maximum = max(data)
    magnitude = len(str(int(maximum))) - 1
    pow_ = 10 ** magnitude
    normalized_decScal = [x / pow_ for x in data]
    return normalized_decScal
def calculate():
    try:
        A = column_select.get()  # Changed from entry.get()
        df[A].fillna(0, inplace=True)
        numbers = df[A].to_list()

        if var5.get():
            bin_size_entry = entry_bin_size.get()
            if bin_size_entry.isdigit():
                bin_size = int(bin_size_entry)
                mean_smoothed = mean_smoothing(numbers, bin_size)
                df[f"{A}_MeanSmoothed"] = mean_smoothed

        if var6.get():
            bin_size_entry = entry_bin_size.get()
            if bin_size_entry.isdigit():
                bin_size = int(bin_size_entry)
                median_smoothed = median_smoothing(numbers, bin_size)
                df[f"{A}_MedianSmoothed"] = median_smoothed

        if var7.get():
            bin_size_entry = entry_bin_size.get()
            if bin_size_entry.isdigit():
                bin_size = int(bin_size_entry)
                boundary_smoothed = boundary_smoothing(numbers, bin_size)
                df[f"{A}_BoundarySmoothed"] = boundary_smoothed

        if var9.get():
            norm = min_max_normalization(numbers)
            df[f"{A}_MinMaxNormalized"] = norm
            print(f"Columns after adding mean smoothed: {df.columns}")


        if var10.get():
            norm = z_score_normalization(numbers)
            df[f"{A}_ZScoreNormalized"] = norm

        if var11.get():
            norm = normalization_by_decimal_scaling(numbers)
            df[f"{A}_DecimalScaled"] = norm

        messagebox.showinfo("Success", "Preprocessing applied and columns added to the original dataset.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
def show_data():
    global filename, df
    if filename:
        try:
            if df.empty:
                messagebox.showwarning("Warning", "File is empty.")
            else:
                second_window = tk.Toplevel()
                second_window.title("Data Set")
    
                tree = ttk.Treeview(second_window, columns=df.columns.tolist(), show="headings")
                for col in df.columns:
                   tree.heading(col, text=col)
                tree.pack()

                rows = df.head().values.tolist()
                for row in rows:
                   tree.insert("", "end", values=row)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showwarning("Warning", "Please upload a file first.")
def show_statistics():
    global filename, df
    if df.empty:
        messagebox.showwarning("Warning", "No data to display statistics.")
        return
    
    statistics_window = tk.Toplevel()
    statistics_window.title("Statistics Description")

    stats_text = tk.Text(statistics_window)
    stats_text.pack()

    stats_text.insert(tk.END, "Statistics Description:\n\n")
    stats_text.insert(tk.END, f"Number of rows: {len(df)}\n")
    stats_text.insert(tk.END, f"Number of columns: {len(df.columns)}\n\n")

    for column in df.select_dtypes(include=np.number).columns:
        
        stats_text.insert(tk.END, f"Column: {column}\n")
        stats_text.insert(tk.END, f"Mean: {statistics.mean(df[column])}\n")
        stats_text.insert(tk.END, f"Median: {statistics.median(df[column])}\n")
        stats_text.insert(tk.END, f"Mode: {statistics.mode(df[column])}\n")
        stats_text.insert(tk.END, f"Standard Deviation: {statistics.stdev(df[column])}\n")
        stats_text.insert(tk.END, f"Variance: {statistics.variance(df[column])}\n\n")
def perform_linear_regression():
    # Get the column names for the independent and dependent variables
    x_var = x_var_select.get()
    y_var = y_var_select.get()
    print(x_var, y_var)
    
    # Reshape x_data and y_data into 2D arrays
    x_data = df[[x_var]].values.reshape(-1, 1)  # Select the column and reshape to 2D
    y_data = df[y_var].values.reshape(-1, 1)  # Select the column and reshape to 2D

    # Perform the linear regression
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train) 
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    plt.scatter(x_train, y_train, color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.title(f"{x_var} vs {y_var} (Training set)") 
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.show()
    plt.scatter(x_test, y_test, color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.title(f"{x_var} vs {y_var} (test set)") 
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.show()


root = tk.Tk()
root.title("Data Preprocessing")

notebook = ttk.Notebook(root)

# Create frames for each tab
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)

# Add the frames as tabs to the notebook
notebook.add(tab1, text='Tab 1')
notebook.add(tab2, text='Tab 2')
button1 = tk.Button(tab1, text="Upload Excel File", command=upload_file)
button1.pack(anchor='n')
# Place the notebook on the window
notebook.pack(expand=True, fill='both')

button_show_rows = tk.Button(tab1, text="Show data set", command=show_data)
button_show_rows.pack(anchor='n')

button_show_statistics = tk.Button(tab1, text="Show Statistics", command=show_statistics)
button_show_statistics.pack()
column_select = tk.StringVar()
column_menu = ttk.Combobox(tab1, textvariable=column_select)
column_menu.pack(anchor='w')


labelsm = tk.Label(tab1, text="----Smoothing----")

labelsm.pack(anchor='w')

var5 = tk.IntVar()
chk5 = tk.Checkbutton(tab1, text='Smoothing by mean', variable=var5)
chk5.pack(anchor='w')

var6 = tk.IntVar()
chk6 = tk.Checkbutton(tab1, text='Smoothing by median', variable=var6)
chk6.pack(anchor='w')

var7 = tk.IntVar()
chk7 = tk.Checkbutton(tab1, text='Smoothing by bounders', variable=var7)
chk7.pack(anchor='w')

label_bin_size = tk.Label(tab1, text="Enter bin size for smoothing:")
label_bin_size.pack(anchor='w')

entry_bin_size = tk.Entry(tab1)
entry_bin_size.pack(anchor='w')

labelno = tk.Label(tab1, text="----Normalization----")
labelno.pack(anchor='w')

var9 = tk.IntVar()
chk9 = tk.Checkbutton(tab1, text='min max normalization', variable=var9)
chk9.pack(anchor='w')

var10 = tk.IntVar()
chk10 = tk.Checkbutton(tab1, text='z score normalization', variable=var10)
chk10.pack(anchor='w')

var11 = tk.IntVar()
chk11 = tk.Checkbutton(tab1, text='normalization by decimal scaling', variable=var11)
chk11.pack(anchor='w')

buttonbox = tk.Button(tab1, text="Box Plot", command=box_plot)
buttonbox.pack(anchor='w')

button = tk.Button(tab1, text="Calculate", command=calculate)
button.pack(anchor='w')



label = tk.Label(tab1, text="Enter name of column:")
label.pack(anchor='w')  


x_var_select = tk.StringVar()
y_var_select = tk.StringVar()
x_var_menu = ttk.Combobox(tab2, textvariable=x_var_select)
x_var_menu.pack(anchor='w')

y_var_menu = ttk.Combobox(tab2, textvariable=y_var_select)
y_var_menu.pack(anchor='w')
linear_regression_button = tk.Button(tab2, text="Perform Linear Regression", command=perform_linear_regression)
linear_regression_button.pack(anchor='w')


root.mainloop()   
