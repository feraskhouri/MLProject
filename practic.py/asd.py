import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Data Preprocessing")

notebook = ttk.Notebook(root)

# Create frames for each tab
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)

# Add the frames as tabs to the notebook
notebook.add(tab1, text='Tab 1')
notebook.add(tab2, text='Tab 2')

# Place the notebook on the window
notebook.pack(expand=True, fill='both')

# Now you can add your widgets to each tab
# For example, let's add a button to Tab 1
button1 = tk.Button(tab1, text="Upload Excel File")
button1.pack(anchor='n')

# And a label to Tab 2
label = tk.Label(tab2, text="Enter name of column:")
label.pack(anchor='w')

root.mainloop()
