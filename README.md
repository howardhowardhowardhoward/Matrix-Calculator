
___________________________________________________________________________________________________________________________________________

### Matrix Calculator

This is a toy matrix calculator designed to perform fundamental matrix operations and display key properties of a given matrix. The tool supports: 

- **Row-Reduced Echelon Form (RREF)**
- **Determinant calculation**
- **Matrix rank and nullity**
- **Matrix inverse (if it exists)**
- **Kernel (null space) computation**
- **Matrix addition and multiplication**
- **Matrix scalar multiplication**
- **Matrix transpose**
 
___________________________________________________________________________________________________________________________________________

### Input Format

Matrices can be entered into a text box using space-separated values for columns and newlines for rows. Examples:

<div style="display: flex; justify-content: space-between;">
  <table>
    <tr><td>0</td><td>1</td><td>2</td></tr>
    <tr><td>23</td><td>4</td><td>5</td></tr>
    <tr><td>6</td><td>7</td><td>8</td></tr>
  </table>

  <table>
    <tr><td>0.5</td><td>1</td></tr>
    <tr><td>2</td><td>3.7</td></tr>
  </table>

  <table>
    <tr><td>1/2</td><td>2</td></tr>
    <tr><td>1/3</td><td>0.4</td></tr>
  </table>
</div>

Both decimal, integer, and fractional inputs are supported.

___________________________________________________________________________________________________________________________________________

### Instalation Instructions

1. Clone this repository:
`git clone https://github.com/howardhowardhowardhoward/Matrix-Project.git`

3. Download required libraries:
`pip install -r requirements.txt`

5. Run the program:
   python GUI.py

___________________________________________________________________________________________________________________________________________

### Project Structure

	- Matrix.py and MatrixOperations.py: Backend logic for matrix representation and operations.
 		- Matrix.py:
   		- MatrixOperations:
 
	•	GUI.py: Graphical user interface for interacting with the calculator.

