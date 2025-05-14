import tkinter
from tkinter import *
from Matrix import *
from MatrixOperations import *

def destroy_window():
    """
    removes all widgets from the window
    """
    for widget in window.winfo_children():
        widget.destroy()

def is_float(value) -> bool:
    """
    Checks if the entered value can be converted into a float
    """
    try:
        Fraction(value)
        return True
    except ValueError:
        return False

def check_dimensions(m: Matrix, dimension: tuple) -> bool:
    """
    Verifies that the dimensions of the matrix are given the dimension requirement.
    """
    rows_match = dimension[0] == -1 or m.dimensions[0] == dimension[0]
    cols_match = dimension[1] == -1 or m.dimensions[1] == dimension[1]
    return rows_match and cols_match

def check_valid_input(text: str) -> bool:
    """
    Checks if the user input is a valid matrix
    """
    ui = text
    if ui.strip() == "":
        return False
    mlist = [num for line in ui.splitlines() for num in line.split()]
    if not all([c.isdigit() or c.isspace() or is_float(c) for c in mlist]):
        return False
    mlist = [[Fraction(num) for num in line.split()] for line in ui.splitlines()]
    if len(mlist) > 0:
        return all([len(sublist) == len(mlist[0]) for sublist in mlist])
    return True


def get_matrix_from_input(text_widget, dimension: tuple) -> Matrix | None:
    """
    Retrieves and returns the matrix which was entered by the use
    """
    ui = text_widget.get('1.0', tkinter.END).strip()
    if check_valid_input(ui):
        mlist = [[Fraction(num) for num in line.split()] for line in ui.splitlines()]
        matr =  Matrix(mlist)
        if check_dimensions(matr, dimension):
            return matr
        return None
    else:
        return None

def scale(m: Matrix, i: str) -> Matrix | None:
    """
    returns a Matrix scaled by the factor i or None if the input is invalid
    """
    try:
        m.scalar_multiplication(Fraction(i.strip()))
        return m
    except ValueError:
        return None

def scaler(m: Matrix):
    """
    Button function to scale the matrix by the user input
    """
    destroy_window()
    title = Label(window, text = "Scale by", font = ('Times New Roman', 20))
    it = tkinter.Text(window, height=1, width=10)
    title.pack()
    it.pack()
    sub = tkinter.Button(window, text = "SUBMIT", command = lambda: retrieve_input(scale(m, it.get('1.0', tkinter.END))))
    sub.pack()

def adder(m: Matrix):
    """
    Button function to add the matrix to the user input
    """
    destroy_window()
    title = Label(window, text = "Add to", font = ('Times New Roman', 20))
    it = tkinter.Text(window, height=12, width=20)
    title.pack()
    it.pack()
    sub = tkinter.Button(window, text="SUBMIT",
                         command=lambda: retrieve_input(MatrixOperations(get_matrix_from_input(it, (m.dimensions[0], m.dimensions[1])), m).add()))
    sub.pack()

def multiplier(m: Matrix):
    """
    Button function to multiply the matrix to the user input
    """
    destroy_window()
    title = Label(window, text="Multiply to", font=('Times New Roman', 20), pady= 10)
    nah = tkinter.Text(window, height=12, width=20, pady = 10)
    title.pack()
    nah.pack()

    blank = tkinter.Label(window, text = "")
    blank.pack()

    button_frame = tkinter.Frame(window)
    left_multiply = tkinter.Button(button_frame, text = "LEFT MULTIPLY", command = lambda: retrieve_input(MatrixOperations(get_matrix_from_input(nah, (-1, m.dimensions[0])), m).multiply()))
    left_multiply.grid(column = 0, row = 0)
    right_multiply = tkinter.Button(button_frame, text = "RIGHT MULTIPLY", command = lambda: retrieve_input(MatrixOperations(m, get_matrix_from_input(nah, (m.dimensions[1], -1))).multiply()))
    right_multiply.grid(column = 1, row = 0)
    button_frame.pack()
    return


def transposer(m:Matrix):
    """
    Button function to transpose the matrix
    """
    m.transpose()
    retrieve_input(m)

def rise(m:Matrix, i:str) -> Matrix | None:
    """
    Helper function for the powers function
    """
    try:
        res = Matrix(m.rows)
        for _ in range(int(i) - 1):
            res = MatrixOperations(res, m).multiply()
        return res
    except ValueError:
        return None

def powers(m:Matrix):
    """
    Button function to raise the matrix to the given power
    """
    destroy_window()
    title = Label(window, text = "Power of", font = ('Times New Roman', 20))
    it = tkinter.Text(window, height=1, width=10)
    title.pack()
    it.pack()
    sub = tkinter.Button(window, text = 'SUBMIT', command = lambda: retrieve_input(rise(m, it.get('1.0', tkinter.END))))
    sub.pack()

def display_actions(m: Matrix):
    """
    displays the possible actions that can be taken on the matrix
    """
    button_frame = tkinter.Frame(window)

    add = tkinter.Button(button_frame, text = "ADD", command= lambda: adder(m))
    scalar = tkinter.Button(button_frame, text = 'SCALE', command = lambda: scaler(m))
    trans = tkinter.Button(button_frame, text = "TRANSPOSE", command = lambda: transposer(m))
    multiply = tkinter.Button(button_frame, text = 'MULTIPLY', command = lambda: multiplier(m))
    power = tkinter.Button(button_frame, text = 'POWER', command = lambda: powers(m))

    scalar.grid(column = 1, row = 0, pady = 20)
    add.grid(column = 0, row = 0, pady = 20)
    trans.grid(column = 2, row = 0, pady = 20)
    if m.dimensions[0] == m.dimensions[1]:
        power.grid(column = 3, row = 0)
    multiply.grid(column = 4, row = 0)

    button_frame.pack()


def display_properties(m: Matrix):
    """
    Displays the properties of the matrix on screen
    """
    properties = [
        f"det: {m.determinant()}",
        f"rank: {m.rank()}",
        f"null: {m.null()}",
    ]
    for prop in properties:
        labels = Label(window, text=prop, font=("Courier", 15))
        labels.pack(padx = (15, 0))

    main_frame = tkinter.Frame(window)

    rref_matrix = RREF(m).rref()
    inverse_matrix = RREF(m).inverse()
    kernel = display_kernel(m.kernel())

    rref_label = tkinter.Label(main_frame, text= "RREF", font=("Courier", 16))
    rref_matrix = tkinter.Label(main_frame, text = str(rref_matrix), font = ("Courier", 20))

    inverse_label = tkinter.Label(main_frame, text = "Inverse", font=("Courier", 16))
    if m.invertible():
        inverse_mat = tkinter.Label(main_frame, text = str(inverse_matrix), font = ("Courier", 20))
    else:
        inverse_mat = tkinter.Label(main_frame, text = "None", font = ("Courier", 16))

    kernel_label = tkinter.Label(main_frame, text = "Kernel", font=("Courier", 16))
    kernel_mat = tkinter.Label(main_frame, text = kernel, font = ("Courier", 16))


    rref_label.grid(row = 0, column = 0, padx= 15, pady = 5)
    inverse_label.grid(row = 0, column = 1, padx= 15, pady = 5)
    kernel_label.grid(row = 0, column = 2, padx= 15)
    rref_matrix.grid(row = 1, column = 0, padx= 10)
    inverse_mat.grid(row = 1, column = 1, padx= 10)
    kernel_mat.grid(row = 1, column = 2, padx = 10)


    main_frame.pack(padx=10, pady=10)

def retrieve_input(matrix: Matrix| None):
    """
    Retrieves and displays the properties and actions which can be taken on the matrix
    """
    if not matrix:
        Label(window, text = "INVALID INPUT").pack()
        return

    destroy_window()
    result = str(matrix)

    title = Label(window, text="Matrix Properties", font=('Times New Roman', 20))
    title.pack()
    blank_line = Label(window, text = "")
    blank_line.pack()
    new_label = Label(window, text = result, font=('Courier', 25), justify='left')
    new_label.pack()
    blank_line = Label(window, text = "")
    blank_line.pack()
    display_properties(matrix)
    display_actions(matrix)

if __name__ == "__main__":
    window = Tk()
    window.title("MATRIX CALCULATOR")
    window.geometry("600x420")

    label = Label(window, text = "Give me a Matrix", font = ('Times New Roman', 20))
    label.pack()

    inpt = tkinter.Text(window, height = 12, width = 20)
    inpt.pack()

    submit = tkinter.Button(window, text = "SUBMIT", command = lambda: retrieve_input(get_matrix_from_input(inpt, (-1, -1))))
    submit.pack()

    window.mainloop()