from Matrix import Matrix
def dot(lst1: list, lst2: list) -> int:
    """returns the dot product of two lists

    Preconditions:
    - lst1 and lst2 contain oonly integers.
    - len(lst1) == len(lst2)
    - both lists are non-empty

    >>> dot([1,2,3], [2,3,4])
    20
    >>> dot([1,1], [1,1])
    2
    >>> dot([0,0], [1,0])
    0
    """
    sm = 0
    for i in range(len(lst1)):
        sm += lst1[i] * lst2[i]
    return sm

class MatrixOperations:
    """
    Class dedicated to operations between two Matrices.
    """
    matrix1: Matrix
    matrix2: Matrix

    def __init__(self, matrix1:Matrix|None, matrix2: Matrix|None) -> None:
        """
        Initializes the two matrices that will be manipulated
        :param matrix1:
        :param matrix2:
        """
        self.matrix1 = matrix1
        self.matrix2 = matrix2

    def multiply(self) -> Matrix|None:
        """
        return the product of matrix1 and matrix2

        Precondition:
        The dimensions are valid for matrix multiplication.

        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m2 = Matrix([[5, 6], [7, 8]])
        >>> op = MatrixOperations(m1, m2)
        >>> result = op.multiply()
        >>> print(result)
        19 22
        43 50
        """
        if not self.matrix1 or not self.matrix2:
            return None
        new = []
        for row in self.matrix1.rows:
            new_gen = []
            for column in self.matrix2.columns:
                new_gen.append(dot(row, column))
            new.append(new_gen)
        return Matrix(new)

    def add(self) -> Matrix|None:
        """
        return the matrix you get from adding Matrix1 and Matrix2
        Precondition:
        Same dimension
        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m2 = Matrix([[5, 6], [7, 8]])
        >>> op = MatrixOperations(m1, m2)
        >>> result = op.add()
        >>> print(result)
        6 8
        10 12
        """
        if not self.matrix1 or not self.matrix2:
            return None
        new = []
        for i in range(len(self.matrix1.rows)):
            row = []
            for j in range(len(self.matrix1.rows[i])):
                row.append(self.matrix1.rows[i][j] + self.matrix2.rows[i][j])
            new.append(row)
        return Matrix(new)


