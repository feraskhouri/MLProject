import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox,ttk
from tkinter import filedialog
from tkinter import *
from tkinter import font
from PIL import ImageTk,Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

filename = ""
df = pd.DataFrame()  
#-----genral purpuse functions-----#

def X_values():
    global X_values_cru
    X_values_cru=box1.curselection()
    X_values.val = [box1.get(idx) for idx in box1.curselection()]
    
    for i in range(len(list(X_values.val))):
        box2.insert(i, X_values.val[i])
        box1.selection_clear(i, END)
    print("Predictors are:"+ str(X_values.val))

def X_values2():
    global X_values_cru
    X_values_cru=box3.curselection()
    X_values.val = [box3.get(idx) for idx in box3.curselection()]
    
    for i in range(len(list(X_values.val))):
        box4.insert(i, X_values.val[i])
        box3.selection_clear(i, END)
    print("Predictors are:"+ str(X_values.val))

def clearBox2():
    del X_values.val
    box2.delete(0,END)

def clearBox4():
    del X_values.val
    box4.delete(0,END)

def nn():
    x=y_var_select.get()
    b = df[x].values
    column_data = pd.Series(b)
    if df[x].dtype == 'int64' or df[x].dtype == 'object':
        num_unique_values = column_data.nunique()
        if num_unique_values==2:
            labelty.config(text="Binary Classification")
        else:
            labelty.config(text="Multi-class Classification")
    elif df[x].dtype == 'float64':
        labelty.config(text="Regression")
    labelty.grid(row=13, column=2,padx=16)

def populate_x_var_listbox():
    box1.delete(0, tk.END)
    box3.delete(0, tk.END)
    for column in df.columns:
        box1.insert(tk.END, column)
        box3.insert(tk.END, column)

def encodingc(b,h):
    x_vars=X_values_cru
    cx=df.iloc[:, list(x_vars)].values
    for i,x in enumerate(b):
        if df[x].dtype == 'object':
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [int(i)])], remainder='passthrough')
            cx= np.array(ct.fit_transform(h))
    return cx

def upload_file():
    global filename,df
    filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.csv")])
    messagebox.showinfo("File Uploaded", f"File {filename} uploaded successfully.")
    df = pd.read_csv(filename)
    column_names = df.columns.tolist()
    column_select.set('')  # clear the current selection
    column_menu['values'] = column_names
    y_var_menu['values'] = column_names
    visMenu1['values'] = column_names
    regMenuX['values'] = column_names
    regMenuY['values'] = column_names
    populate_x_var_listbox()
    page1.tkraise()

#-----------------------Page1---------------------#

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

def show_statistics():
    import statistics
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

def calculate():
    global df
    try:
        A = column_select.get()
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

        if var110.get():
            column_to_drop = A
            df = df.drop(columns=[column_to_drop])

        messagebox.showinfo("Success", "Preprocessing applied and columns added to dataset.")
        column_names = df.columns.tolist()
        column_select.set('')  
        column_menu['values'] = column_names
        visMenu1['values'] = column_names
        regMenuX['values'] = column_names
        regMenuY['values'] = column_names
        populate_x_var_listbox()
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
                populate_x_var_listbox()

        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showwarning("Warning", "Please upload a file first.")

#-----------------------data vis fun-------------------#

def box_plot():
    A = visMenu1.get()  # Changed from entry.get()    df[A].fillna(0, inplace=True)
    numbers = df[A].to_list()
    fig, ax = plt.subplots()
    ax.boxplot(numbers)
    ax.set_title(f'Box Plot of {A}')
    plt.show()

def scatter_plot():
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import seaborn as sns
    selected_variable = visMenu1.get()
    ax = sns.pairplot(df, hue=selected_variable)
    plt.show()

def voilin_plot():
    import seaborn as sns
    selected_variable = visMenu1.get()
    sns.violinplot(x=selected_variable, y=selected_variable, data=df)
    plt.show()

def Histogram():
    import seaborn as sns
    selected_variable = visMenu1.get()
    sns.histplot(df[selected_variable], kde=True)
    plt.show()    

def heat_map():
    # Convert non-numeric columns to numeric
    import seaborn as sns
    
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    sns.heatmap(df_numeric.corr(), annot=True)
    plt.show()

def Linear_RegressionGraph():
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Ensure regMenuX and regMenuY are not empty and are valid column names in df
    if regMenuX.get() not in df.columns or regMenuY.get() not in df.columns:
        raise ValueError("regMenuX and regMenuY must be valid column names in df")

    # Now you can safely select the data from the DataFrame
    data = df[[regMenuX.get(), regMenuY.get()]]  # Add this line to select the data from the DataFrame

    sns.regplot(x=regMenuX.get(), y=regMenuY.get(), data=data, fit_reg=True, scatter_kws={"color": "#a9a799"}, line_kws={"color": "#835656"})  # Pass the data parameter
    plt.title(f"{regMenuX.get()} vs {regMenuY.get()}")
    plt.show()

def bar_plot():
    import matplotlib.pyplot as plt
    import seaborn as sns
    selected_variable = visMenu1.get()
    sns.barplot(x=selected_variable, y=selected_variable, data=df)
    plt.show()

#---------------ML S----------#

def SVM():
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.preprocessing import MinMaxScaler

    x_vars = X_values_cru
    y_var = y_var_select.get()
    test_var = float(test_entry.get())
    rand_var = int(rand_entry.get())
    kernel = kernelVar.get()
    use_scaling = scale_var.get()
    X = df.iloc[:, list(x_vars)].values
    y = df[y_var].values
    X = encodingc(X_values.val, X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_var, random_state=rand_var)
    if use_scaling:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    classifier = SVC(kernel=kernel, random_state=rand_var)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'The Accuracy: {accuracy:.2f}\nConfusion Matrix')
    plt.show()

    plt.figure(figsize=(10, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.coolwarm)
    if kernel == 'linear':
        w = classifier.coef_[0]
        b = classifier.intercept_[0]
        x_hyperplane = np.linspace(x_min, x_max, 100)
        y_hyperplane = -(w[0] * x_hyperplane + b) / w[1]
        margin = 1 / np.sqrt(np.sum(w ** 2))
        y_margin_pos = y_hyperplane + margin / w[1]
        y_margin_neg = y_hyperplane - margin / w[1]
        plt.plot(x_hyperplane, y_hyperplane, 'k-', label='Decision boundary')
        plt.plot(x_hyperplane, y_margin_pos, 'k--', label='Positive margin')
        plt.plot(x_hyperplane, y_margin_neg, 'k--', label='Negative margin')
    else:
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100,
                facecolors='none', edgecolors='k', label='Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary and Support Vectors')
    unique_labels = np.unique(y)
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles + [plt.Line2D([0], [0], linestyle='-', color='k'),
                                  plt.Line2D([0], [0], linestyle='--', color='k'),
                                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='k', markersize=10)],
               labels=list(unique_labels) + ['Decision boundary', 'Margin', 'Support Vectors'],
               title="Classes")
    plt.show()

def simple_linear_regression():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import  mean_squared_error
    x_var=X_values.val[0]
    y_var = y_var_select.get()
    X = df[[x_var]].values.reshape(-1, 1)
    y = df[y_var].values.reshape(-1, 1)
    test_var = float(test_entry.get())
    rand_var = int(rand_entry.get())
    use_scaling = scale_var.get()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_var, random_state=rand_var)
    if use_scaling:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    accuracy = mean_squared_error(y_test, y_pred)
    plt.scatter(x_train, y_train, color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.title(f"The mse: {accuracy:.2f}\n{x_var} vs {y_var} (Training set)") 
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.show()
    plt.scatter(x_test, y_test, color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.title(f"{x_var} vs {y_var} (test set)") 
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.show()

def multi_lin_reg():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import  mean_squared_error
    x_vars = X_values_cru
    y_var = y_var_select.get()
    X = df.iloc[:, list(x_vars)].values
    y = df[y_var].values
    test_var = float(test_entry.get())
    rand_var = int(rand_entry.get())
    X=encodingc(X_values.val,X)
    use_scaling = scale_var.get()
    if use_scaling:
        sc=StandardScaler()
        X = np.array(sc.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_var, random_state = rand_var)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = mean_squared_error(y_test, y_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolor='w', alpha=0.7, s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'The mse: {accuracy:.2f}\nActual vs Predicted Values')
    plt.show()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

def logistic():
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc

    x_vars = X_values_cru
    y_var = y_var_select.get()
    X = df.iloc[:, list(x_vars)].values
    y = df[y_var].values
    test_var = float(test_entry.get())
    rand_var = int(rand_entry.get())
    X=encodingc(X_values.val,X) 
    use_scaling = scale_var.get()
    if use_scaling:
        sc=StandardScaler()
        X = np.array(sc.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_var, random_state=rand_var)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'The Accuracy: {accuracy:.2f}\nConfusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def KNN():
    from sklearn.neighbors import KNeighborsClassifier

    x_vars = X_values_cru
    y_var = y_var_select.get()
    k = int(k_entry.get())
    X = df.iloc[:, list(x_vars)].values
    y = df[y_var].values
    test_var = float(test_entry.get())
    rand_var = int(rand_entry.get())
    X=encodingc(X_values.val,X) 
    use_scaling = scale_var.get()
    if use_scaling:
        sc=StandardScaler()
        X = np.array(sc.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_var, random_state=rand_var)

    classifier = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'The Accuracy: {accuracy:.2f}\nConfusion Matrix')
    plt.show()

    if X.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.coolwarm)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('KNN Decision Boundary')

        unique_labels = np.unique(y)
        handles, _ = scatter.legend_elements()
        plt.legend(handles=handles, labels=list(unique_labels), title="Classes")
        plt.show()
    else:
        print("Decision boundary visualization is only available for 2D feature spaces.")

def KNN_regression():
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error

    x_vars = X_values_cru
    y_var = y_var_select.get()
    k = int(kr_entry.get())
    X = df.iloc[:, list(x_vars)].values
    y = df[y_var].values
    test_var = float(test_entry.get())
    rand_var = int(rand_entry.get())
    X = encodingc(X_values.val, X)
    use_scaling = scale_var.get()
    if use_scaling:
        sc = StandardScaler()
        X = np.array(sc.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_var, random_state=rand_var)

    regressor = KNeighborsRegressor(n_neighbors=k, metric='minkowski', p=2)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(f'mse: {mse:.2f}\nActual vs Predicted values')
    plt.show()

    # Visualize decision boundary if there are only 2 features
    if X.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.coolwarm)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('KNN Regression Decision Boundary')

        plt.show()
    else:
        print("Decision boundary visualization is only available for 2D feature spaces.")

def SVR_regression():
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error

    x_vars = X_values_cru
    y_var = y_var_select.get()
    test_var = float(test_entry.get())
    rand_var = int(rand_entry.get())
    kernel = kernelVarr.get()
    X = df.iloc[:, list(x_vars)].values
    y = df[y_var].values
    X = encodingc(X_values.val, X)
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_var, random_state=rand_var)

    regressor = SVR(kernel=kernel)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_pred = y_pred.reshape(-1, 1)
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    mse = mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_inverse)

    plt.figure(figsize=(10, 6))
    plt.scatter(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_inverse, color='blue')
    plt.plot([min(scaler_y.inverse_transform(y_test.reshape(-1, 1))), max(scaler_y.inverse_transform(y_test.reshape(-1, 1)))], 
         [min(scaler_y.inverse_transform(y_test.reshape(-1, 1))), max(scaler_y.inverse_transform(y_test.reshape(-1, 1)))], 
         color='red', linewidth=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(f'Mean Squared Error: {mse:.2f}\nActual vs Predicted values')
    plt.show()

    # Visualize decision boundary if there are only 2 features
    if X.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = scaler_y.inverse_transform(Z.reshape(-1, 1)).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.coolwarm)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('SVR Decision Boundary')

        unique_labels = np.unique(y)
        handles, _ = scatter.legend_elements()
        plt.legend(handles=handles, labels=list(unique_labels), title="Classes")
        plt.show()
    else:
        messagebox.showerror("Error","Decision boundary visualization is only available for 2D feature spaces.")

#---------------ML unS----------#

def Kmeans():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    x_vars = X_values_cru
    cmapk=clickedKCMAP.get()
    X = df.iloc[:, list(x_vars)].values
    k=int(km_entry.get())
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_

    pca = PCA(n_components=3)
    reduced_X = pca.fit_transform(X)
    messagebox.showinfo("Success", "KMeans clustering applied and clusters added to the original dataset.")
    # Visualizing the clusters
    plt.scatter(reduced_X[:, 0], reduced_X[:, 0], c=df['Cluster'], cmap=cmapk)
    plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 0], marker='.', s=300, c='red', label='Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering with DataFrame')
    plt.legend()
    plt.show()

def dbscan():
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    x_vars = X_values_cru
    #print(x_vars)
    cmap= optionscmap.get()
    c=int(db_entry.get())
    epsEntry = float(eps_entry.get())
    X = df.iloc[:, list(x_vars)].values
    dbscan = DBSCAN(eps=epsEntry, min_samples=c)
    labels = dbscan.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering')
    plt.show()
def hca():
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import AgglomerativeClustering
    try:
        # Retrieve and validate the number of clusters from the entry widget
        c = hca_entry.get()
        if not c.strip():
            raise ValueError("Number of clusters is required.")
        c = int(c)
        metric = clicked.get()
        print(metric)
        linkage = clickedLinkage.get()
        print(linkage)
        x_vars = X_values_cru  
        X = df.iloc[:, list(x_vars)].values  # Assuming df is your DataFrame

        # Perform Hierarchical Clustering
        hc = AgglomerativeClustering(n_clusters=c, metric=metric, linkage=linkage)
        y_hc = hc.fit_predict(X)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, y_hc)
        tk.messagebox.showinfo("Silhouette Score", f'Silhouette Score: {silhouette_avg:.2f}')

        # Plotting the clusters
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange']

        if X.shape[1] == 1:
            # One-dimensional data
            plt.figure(figsize=(10, 6))
            for i in range(c):
                plt.scatter(X[y_hc == i], np.zeros_like(X[y_hc == i]), s=100, c=colors[i % len(colors)], label=f'Cluster {i+1}')
            plt.title('Clusters of customers (1D)')
            plt.xlabel('Feature')
            plt.yticks([])
        elif X.shape[1] >= 2:
            # Two-dimensional data
            plt.figure(figsize=(10, 6))
            for i in range(c):
                plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=colors[i % len(colors)], label=f'Cluster {i+1}')
            plt.title('Clusters of customers (2D)')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')

        plt.legend()
        plt.show()

    except ValueError as ve:
        # Show an error message if the entry is invalid
        messagebox.showerror("Invalid Input", str(ve))
    except IndexError as ie:
        # Show an error message if there is an index error
        messagebox.showerror("Index Error", str(ie))



def deter_hca():
    import scipy.cluster.hierarchy as sch
    x_vars = X_values_cru
    X = df.iloc[:, list(x_vars)].values
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

def deter_k():
    from sklearn.cluster import KMeans
    x_vars = X_values_cru
    X = df.iloc[:, list(x_vars)].values
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

#-----------------------root-------------------#
root = Tk()
root.title("DML")
root.geometry('700x600')
root.resizable(False,False)
page0=Frame(root)
page1=Frame(root)
page2=Frame(root)
page3=Frame(root)
page4=Frame(root)
page0.grid(row=0,column=0,sticky="nsew")
page1.grid(row=0,column=0,sticky="nsew")
page2.grid(row=0,column=0,sticky="nsew")
page3.grid(row=0,column=0,sticky="nsew")
page4.grid(row=0,column=0,sticky="nsew")
sty1=font.Font(size=18,slant="italic",underline=1)
sty2=font.Font(size=12)
sty3=font.Font(size=15,slant="italic",underline=1)
sty4=font.Font(size=11)
sty5=font.Font(size=12,slant="italic",underline=1)
sty6=font.Font(size=10)
page0.tkraise()
#----------------------------page0-------------------------------#


labelintro = tk.Label(page0, text="Introduction",font=sty1)
labelintro.grid(row=1, column=0, padx=10, pady=0)

labeltext = tk.Label(page0, text="First of all, we thank Dr. Mohamed Almsedin\n for putting his trust in us and giving us knowledge to write this program\n, which we hope you will like and appreciate.\n We trust that it will assist you in performing various functions.\n In short, this program encompasses pre-processing \napplications in data mining and several machine learning algorithms.\n Therefore, we have named it DML, which stands for Data Mining - Machine Learning.",font=sty2)
labeltext.grid(row=2, column=0, padx=10, pady=15)

button_up = tk.Button(page0, text="Upload CSV file",font=sty2, command=upload_file)
button_up.grid(row=3, column=0, padx=0, pady=30)

button0t1 = tk.Button(page0, text="->",font=sty6, command=lambda: page1.tkraise())
button0t1.grid(row=5, column=0, padx=0, pady=15)
#----------------------------page1-----------------------------#

labelp2 = tk.Label(page1, text="Pre-processing section",font=sty3)
labelp2.grid(row=0, column=1, padx=0, pady=15)

button_show_rows = tk.Button(page1, text="Show data set",font=sty2, command=show_data)
button_show_rows.grid(row=1, column=0, padx=70, pady=30)

button_show_statistics = tk.Button(page1, text="Show Statistics",font=sty2, command=show_statistics)
button_show_statistics.grid(row=1, column=2, padx=45, pady=15)

label = tk.Label(page1, text="Select the column:",font=sty4)
label.grid(row=2, column=1, padx=0, pady=0)

column_select = tk.StringVar()
column_menu = ttk.Combobox(page1, textvariable=column_select,font=sty2)
column_menu.grid(row=3, column=1, padx=0, pady=15)

labelsm = tk.Label(page1, text="Smoothing",font=sty5)
labelsm.grid(row=4, column=0, padx=23, pady=15)

labelno = tk.Label(page1, text="Normalization",font=sty5)
labelno.grid(row=4, column=2, padx=23, pady=15)

var5 = tk.IntVar()
chk5 = tk.Checkbutton(page1, text='Smoothing by mean      ',font=sty6, variable=var5)
chk5.grid(row=5, column=0, padx=0, pady=5)

var9 = tk.IntVar()
chk9 = tk.Checkbutton(page1, text='min max normalization        ',font=sty6, variable=var9)
chk9.grid(row=5, column=2, padx=15, pady=5)

var6 = tk.IntVar()
chk6 = tk.Checkbutton(page1, text='Smoothing by median   ',font=sty6, variable=var6)
chk6.grid(row=6, column=0, padx=0, pady=5)

var10 = tk.IntVar()
chk10 = tk.Checkbutton(page1, text='z score normalization          ',font=sty6, variable=var10)
chk10.grid(row=6, column=2, padx=15, pady=5)

var7 = tk.IntVar()
chk7 = tk.Checkbutton(page1, text='Smoothing by bounders',font=sty6, variable=var7)
chk7.grid(row=7, column=0, padx=0, pady=7)

var11 = tk.IntVar()
chk11 = tk.Checkbutton(page1, text='normalization by decimal sc ',font=sty6, variable=var11)
chk11.grid(row=7, column=2, padx=20, pady=5)

label_bin_size = tk.Label(page1, text="Enter bin size:",font=sty6)
label_bin_size.grid(row=8, column=0, padx=0, pady=0)

entry_bin_size = tk.Entry(page1)
entry_bin_size.grid(row=9, column=0, padx=0, pady=0)

labelbo = tk.Label(page1, text="Droping col",font=sty5)
labelbo.grid(row=5, column=1, padx=0, pady=0)
var110 = tk.IntVar()
chk110 = tk.Checkbutton(page1, text='drop the col selected',font=sty6, variable=var110)
chk110.grid(row=6, column=1, padx=0, pady=5)

button = tk.Button(page1, text="Calculate",font=sty2, command=calculate)
button.grid(row=13, column=1, padx=0, pady=30)

button1t0 = tk.Button(page1, text="<-",font=sty6, command=lambda: page0.tkraise())
button1t0.grid(row=15, column=0, padx=0, pady=20)

button1t2 = tk.Button(page1, text="->",font=sty6, command=lambda: page2.tkraise())
button1t2.grid(row=15, column=2, padx=0, pady=20)
#----------------------------page2-----------------------------#

labelp2 = tk.Label(page2, text="data visualiztion section",font=sty3)
labelp2.grid(row=0, column=1, padx=0, pady=15)


button_showScatterPlot = tk.Button(page2, text="Show Scatter plot",font=sty2, command=scatter_plot)
button_showScatterPlot.grid(row=2, column=0, padx=70, pady=0)

button_showBarPlot = tk.Button(page2, text="Show Bar plot",font=sty2, command=bar_plot)
button_showBarPlot.grid(row=3, column=0, padx=0, pady=0)
buttonbox = tk.Button(page2, text="Box Plot",font=sty4, command=box_plot)
buttonbox.grid(row=4, column=0, padx=0, pady=5)

buttonbox = tk.Button(page2, text="Histogram",font=sty4, command=Histogram)
buttonbox.grid(row=5, column=0, padx=0, pady=5)

buttonbox = tk.Button(page2, text="volion plot",font=sty4, command=voilin_plot)
buttonbox.grid(row=6, column=0, padx=0, pady=5)


visMenu1 = tk.StringVar()
visMenu1 = ttk.Combobox(page2, textvariable=visMenu1,font=sty2)
visMenu1.grid(row=7, column=0, padx=0, pady=15)



button_showHeatMap = tk.Button(page2, text="Show Heat Map",font=sty2, command=heat_map)
button_showHeatMap.grid(row=1, column=1, padx=70, pady=30)


regMenuX = tk.StringVar()
regMenuX = ttk.Combobox(page2, textvariable=regMenuX,font=sty2)
regMenuX.grid(row=3, column=2, padx=10, pady=15)

regMenuY = tk.StringVar()
regMenuY = ttk.Combobox(page2, textvariable=regMenuY,font=sty2)
regMenuY.grid(row=4, column=2, padx=10, pady=15)


ButtonregLine = tk.Button(page2, text="Linear Regression",font=sty2,textvariable=regMenuX and regMenuY ,command=Linear_RegressionGraph)
ButtonregLine.grid(row=2, column=2, padx=10, pady=15)

button2t1 = tk.Button(page2, text="<-",font=sty6, command=lambda: page1.tkraise())
button2t1.grid(row=15, column=0, padx=0, pady=15)

button2t3 = tk.Button(page2, text="->",font=sty6, command=lambda: page3.tkraise())
button2t3.grid(row=15, column=2, padx=0, pady=20)

#-------------------------page3-----------------------#

LabelFrame1=LabelFrame(page3,text="Loading data")
LabelFrame1.pack(anchor='w',ipady=5)

col_label = tk.Label(LabelFrame1, text="Select Independent Variables (X):")
col_label.grid(row=9, column=0,padx=30)

x_var_label = tk.Label(LabelFrame1, text="Selected Independent Variables (X):")
x_var_label.grid(row=9, column=1,padx=30)

box1 = Listbox(LabelFrame1,selectmode='multiple', exportselection=0)
box1.grid(row=10, column=0,padx=30)

box2 = Listbox(LabelFrame1)
box2.grid(row=10, column=1,padx=30)

Button(LabelFrame1, text='Select Predictors',command=X_values).grid(row=12,column=1)
Button(LabelFrame1, text='Clear Predictors',command=clearBox2).grid(row=13,column=1)

y_var_select = tk.StringVar()
y_var_label = tk.Label(LabelFrame1, text="Select Dependent Variable (Y):")
y_var_label.grid(row=9, column=2,padx=26)

y_var_menu = ttk.Combobox(LabelFrame1, textvariable=y_var_select)
y_var_menu.grid(row=10, column=2,padx=30)

in_button = tk.Button(LabelFrame1, text="Type of supervised learning", command=nn)
in_button.grid(row=12, column=2,padx=20)
labelty = tk.Label(LabelFrame1, text="-",font=sty5)
labelty.grid_forget()

frameins=LabelFrame(page3,text="Parameters")
frameins.pack(anchor='w',padx=75,ipady=5)
test_label = tk.Label(frameins, text="Enter test size:")
test_label.grid(row=1, column=0,padx=30)
test_entry = tk.Entry(frameins)
test_entry.insert(0, "0.2")
test_entry.grid(row=2, column=0,padx=30)
rand_label = tk.Label(frameins, text="Enter random state:")
rand_label.grid(row=1, column=1,padx=30)
rand_entry = tk.Entry(frameins)
rand_entry.insert(0, "0")
rand_entry.grid(row=2, column=1,padx=30)
scale_var = tk.BooleanVar()
scale_checkbox = tk.Checkbutton(frameins, text="Use Standard Scaling", variable=scale_var)
scale_checkbox.grid(row=2, column=2,padx=30)

container = tk.Frame(page3)
container.pack(anchor='w', pady=5, padx=20, ipady=0)

Frame1=LabelFrame(container,text="Linear Regression algo")
Frame1.pack(side='left', padx=5, pady=0,ipady=5)

lin_button = tk.Button(Frame1, text="Simple Linear Regression", command=simple_linear_regression)
lin_button.grid(row=0, column=0,padx=30,pady=15)

mlin_button = tk.Button(Frame1, text="Multiple Linear Regression", command=multi_lin_reg)
mlin_button.grid(row=3, column=0,padx=30,pady=25)

Frame2=LabelFrame(container,text="Classification")
Frame2.pack(side='left', padx=10, pady=0)

log_button = tk.Button(Frame2, text="Logistic regression", command=logistic)
log_button.grid(row=1, column=0,padx=30)

k_label = tk.Label(Frame2, text="Enter the value of K:")
k_label.grid(row=2, column=0,padx=30)

k_entry = tk.Entry(Frame2)
k_entry.grid(row=3, column=0,padx=30)

knn_button = tk.Button(Frame2, text="K-Nearest Neighbors", command=KNN)
knn_button.grid(row=4, column=0,padx=30)

OPTIONS = [
"Select model",
"linear",
"poly",
"rbf"
]
kernelVar = StringVar()
kernelVar.set(OPTIONS[0])
ker_label = tk.Label(Frame2, text="Choose kernel:")
ker_label.grid(row=5, column=0,padx=30,pady=0)
kernelFunc=OptionMenu(Frame2, kernelVar, *OPTIONS)
kernelFunc.grid(row=6,column=0,pady=0)

svm_button = tk.Button(Frame2, text="Support vector machine", command=SVM)
svm_button.grid(row=7, column=0,padx=30,pady=7)


Frame3=LabelFrame(container,text="Regression")
Frame3.pack(side='left', padx=10, pady=0)

kr_label = tk.Label(Frame3, text="Enter the value of K:")
kr_label.grid(row=1, column=0,padx=30)

kr_entry = tk.Entry(Frame3)
kr_entry.grid(row=2, column=0,padx=30)

knnr_button = tk.Button(Frame3, text="K-Nearest Neighbors", command=KNN_regression)
knnr_button.grid(row=4, column=0,padx=30)

kernelVarr = StringVar()
kernelVarr.set(OPTIONS[0])
kerr_label = tk.Label(Frame3, text="Choose kernel:")
kerr_label.grid(row=5, column=0,padx=30,pady=0)
kernelFuncr=OptionMenu(Frame3, kernelVarr, *OPTIONS)
kernelFuncr.grid(row=6,column=0,pady=0)

svmr_button = tk.Button(Frame3, text="Support vector machine", command=SVR_regression)
svmr_button.grid(row=7, column=0,padx=30,pady=7)


containerback = tk.Frame(page3)
containerback.pack(anchor='w', pady=0, padx=20, ipadx=165)

button3t2 = tk.Button(containerback, text="<-",font=sty6, command=lambda: page2.tkraise())
button3t2.pack(side='left',anchor='w',padx=97,pady=0)

button3t4 = tk.Button(containerback, text="->",font=sty6, command=lambda: page4.tkraise())
button3t4.pack(side='right',anchor='n',pady=0)

# ----------------------------page4-----------------------------#

# Loading data section at the top
LabelFrame14 = LabelFrame(page4, text="Loading data")
LabelFrame14.grid(row=0, column=0, ipadx=5, ipady=5, padx=10, pady=10)

col_label = tk.Label(LabelFrame14, text="Select Independent Variables (X):")
col_label.grid(row=0, column=0, padx=30, pady=5)

x_var_label = tk.Label(LabelFrame14, text="Selected Independent Variables (X):")
x_var_label.grid(row=0, column=1, padx=30, pady=5)

box3 = tk.Listbox(LabelFrame14, selectmode='multiple', exportselection=0)
box3.grid(row=1, column=0, padx=30, pady=5)

box4 = tk.Listbox(LabelFrame14)
box4.grid(row=1, column=1, padx=30, pady=5)

tk.Button(LabelFrame14, text='Select Predictors',command=X_values2).grid(row=2, column=1, pady=5)
tk.Button(LabelFrame14, text='Clear Predictors', command=clearBox4).grid(row=3, column=1, pady=5)


# Clustering data section
LabelFrame140 = LabelFrame(page4, text="Clustering data using KMeans, HCA, and DB-Scan")
LabelFrame140.grid(row=2, column=0, padx=10, pady=1)

# KMeans Section
kmeans_frame = LabelFrame(LabelFrame140, text="KMeans")
kmeans_frame.grid(row=0, column=0, padx=5, pady=1)

dk_button = tk.Button(kmeans_frame, text="Determine K", command=deter_k)
dk_button.grid(row=0, column=0, padx=0, pady=1)

km_label = tk.Label(kmeans_frame, text="Enter K:")
km_label.grid(row=1, column=0, padx=5, pady=1)

km_entry = tk.Entry(kmeans_frame)
km_entry.grid(row=1, column=1, padx=5, pady=1)

optionsKCMAP = ["viridis", "plasma", "inferno", "magma"]
clickedKCMAP = StringVar()
clickedKCMAP.set("viridis")
drop = OptionMenu(kmeans_frame, clickedKCMAP, *optionsKCMAP)
drop.grid(row=3, column=0, columnspan=2, padx=5, pady=1)

km_button = tk.Button(kmeans_frame, text="KMeans", command=Kmeans)
km_button.grid(row=4, column=0, columnspan=2, padx=5, pady=1)

# HCA Section
hca_frame = LabelFrame(LabelFrame140, text="HCA")
hca_frame.grid(row=0, column=1, padx=5, pady=1)

dchca_button = tk.Button(hca_frame, text="Determine clusters", command=deter_hca)
dchca_button.grid(row=1, column=0, padx=0, pady=1)


hca_label = tk.Label(hca_frame, text="Enter clusters:")
hca_label.grid(row=2, column=0, padx=5, pady=1)

hca_entry = tk.Entry(hca_frame)
hca_entry.grid(row=2, column=1, padx=5, pady=1)

options = ["euclidean", "manhattan", "jaccard", "cosine", "minkowski"]
clicked = StringVar()
clicked.set("euclidean")
drop = OptionMenu(hca_frame, clicked, *options)
drop.grid(row=4, column=0, columnspan=2, padx=5, pady=1)

optionsLinkage = ["ward", "complete", "average", "single"]
clickedLinkage = StringVar()
clickedLinkage.set("ward")
dropLinkage = OptionMenu(hca_frame, clickedLinkage, *optionsLinkage)
dropLinkage.grid(row=3, column=0, columnspan=2, padx=5, pady=1)

hca_button = tk.Button(hca_frame, text="HCA", command=hca)
hca_button.grid(row=5, column=0, columnspan=2, padx=5, pady=1)

# DB-Scan Section
dbscan_frame = LabelFrame(LabelFrame140, text="DB-Scan")
dbscan_frame.grid(row=0, column=2, padx=5, pady=1)

db_label = tk.Label(dbscan_frame, text="number of min point:")
db_label.grid(row=0, column=0, padx=5, pady=1)

db_entry = tk.Entry(dbscan_frame)
db_entry.grid(row=0, column=1, padx=5, pady=1)

eps_label = tk.Label(dbscan_frame, text="Enter eps:")
eps_label.grid(row=1, column=0, padx=5, pady=1)

eps_entry = tk.Entry(dbscan_frame)
eps_entry.grid(row=1, column=1, padx=5, pady=1)

optionscmap = ["viridis", "plasma", "inferno", "magma"]
clickedcmap = StringVar()
clickedcmap.set("viridis")
dropcmap = OptionMenu(dbscan_frame, clickedcmap, *optionscmap)
dropcmap.grid(row=2, column=0, columnspan=2, padx=5, pady=1)

db_button = tk.Button(dbscan_frame, text="DB-Scan", command=dbscan)
db_button.grid(row=3, column=0, columnspan=2, padx=5, pady=1)

button4t3 = tk.Button(page4, text="<-", font=sty6, command=lambda: page3.tkraise())
button4t3.grid(row=5, column=0, padx=0, pady=1)

root.mainloop()