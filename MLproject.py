import statistics
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import messagebox,ttk
from tkinter import filedialog
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from tkinter import *
from tkinter import font
from PIL import ImageTk,Image



filename = ""
df = pd.DataFrame()  

def KNN(x,y,k,ss):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    if ss:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
def upload_file():
    global filename,df
    filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.csv")])
    messagebox.showinfo("File Uploaded", f"File {filename} uploaded successfully.")
    df = pd.read_csv(filename)
    column_names = df.columns.tolist()
    column_select.set('')  # clear the current selection
    column_menu['values'] = column_names
    x_var_menu['values'] = column_names
    y_var_menu['values'] = column_names
    page2.tkraise()
def box_plot():
    A = column_select.get()  # Changed from entry.get()    df[A].fillna(0, inplace=True)
    numbers = df[A].to_list()
    fig, ax = plt.subplots()
    ax.boxplot(numbers)
    ax.set_title(f'Box Plot of {A}')
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
    minimum = min(data)
    magnitude1 = len(str(int(maximum))) - 1
    magnitude2 = len(str(int(minimum))) - 1
    j=max(magnitude1,magnitude2)
    pow_ = 10 ** j
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

#-----------------------root-------------------#
root = Tk()
root.iconbitmap("D:\MLProject\DML.jpg")
root.title("DML")
root.geometry('700x600')
root.resizable(False,False)

page1=Frame(root)
page2=Frame(root)
page3=Frame(root)
page1.grid(row=0,column=0,sticky="nsew")
page2.grid(row=0,column=0,sticky="nsew")
page3.grid(row=0,column=0,sticky="nsew")

sty1=font.Font(size=18,slant="italic",underline=1)
sty2=font.Font(size=12)
page1.tkraise()
#----------------------------page1-------------------------------#
logopath = Image.open("D:\MLProject\DML.jpg")
resize_dml = logopath.resize((400, 150))
dml_logo = ImageTk.PhotoImage(resize_dml)
dml_label = Label(page1, image=dml_logo)
dml_label.grid(row=0, column=0, padx=150, pady=45)

labelintro = tk.Label(page1, text="Introduction",font=sty1)
labelintro.grid(row=1, column=0, padx=10, pady=0)

labeltext = tk.Label(page1, text="First of all, we thank Dr. Mohamed Almsedin\n for putting his trust in us and helping us write this program\n, which we hope you will like and appreciate.\n We trust that it will assist you in performing various functions.\n In short, this program encompasses pre-processing \napplications in data mining and several machine learning algorithms.\n Therefore, we have named it DML, which stands for Data Mining - Machine Learning.",font=sty2)
labeltext.grid(row=2, column=0, padx=10, pady=15)

button_up = tk.Button(page1, text="Upload CSV file",font=sty2, command=upload_file)
button_up.grid(row=3, column=0, padx=0, pady=30)

button1t2 = tk.Button(page1, text="->", command=lambda: page2.tkraise())
button1t2.grid(row=5, column=0, padx=0, pady=15)

#----------------------------page2-----------------------------#

button_show_rows = tk.Button(page2, text="Show data set", command=show_data)
button_show_rows.pack(anchor='n')

button_show_statistics = tk.Button(page2, text="Show Statistics", command=show_statistics)
button_show_statistics.pack()

label = tk.Label(page2, text="Select the column:")
label.pack(anchor='w')

column_select = tk.StringVar()
column_menu = ttk.Combobox(page2, textvariable=column_select)
column_menu.pack(anchor='w')

labelsm = tk.Label(page2, text="----Smoothing----")
labelsm.pack(anchor='w')

var5 = tk.IntVar()
chk5 = tk.Checkbutton(page2, text='Smoothing by mean', variable=var5)
chk5.pack(anchor='w')

var6 = tk.IntVar()
chk6 = tk.Checkbutton(page2, text='Smoothing by median', variable=var6)
chk6.pack(anchor='w')

var7 = tk.IntVar()
chk7 = tk.Checkbutton(page2, text='Smoothing by bounders', variable=var7)
chk7.pack(anchor='w')

label_bin_size = tk.Label(page2, text="Enter bin size for smoothing:")
label_bin_size.pack(anchor='w')

entry_bin_size = tk.Entry(page2)
entry_bin_size.pack(anchor='w')

labelno = tk.Label(page2, text="----Normalization----")
labelno.pack(anchor='w')

var9 = tk.IntVar()
chk9 = tk.Checkbutton(page2, text='min max normalization', variable=var9)
chk9.pack(anchor='w')

var10 = tk.IntVar()
chk10 = tk.Checkbutton(page2, text='z score normalization', variable=var10)
chk10.pack(anchor='w')

var11 = tk.IntVar()
chk11 = tk.Checkbutton(page2, text='normalization by decimal scaling', variable=var11)
chk11.pack(anchor='w')

buttonbox = tk.Button(page2, text="Box Plot", command=box_plot)
buttonbox.pack(anchor='w')

button = tk.Button(page2, text="Calculate", command=calculate)
button.pack(anchor='w')

button2t1 = tk.Button(page2, text="<-", command=lambda: page1.tkraise())
button2t1.pack(anchor='n')

button2t3 = tk.Button(page2, text="->", command=lambda: page3.tkraise())
button2t3.pack(anchor='n')

#-------------------------page3-----------------------#

x_var_select = tk.StringVar()
y_var_select = tk.StringVar()
x_var_menu = ttk.Combobox(page3, textvariable=x_var_select)
x_var_menu.pack(anchor='w')

y_var_menu = ttk.Combobox(page3, textvariable=y_var_select)
y_var_menu.pack(anchor='w')
linear_regression_button = tk.Button(page3, text="Perform Linear Regression", command=perform_linear_regression)
linear_regression_button.pack(anchor='w')

button3t2 = tk.Button(page3, text="<-", command=lambda: page2.tkraise())
button3t2.pack(anchor='n')


root.mainloop()