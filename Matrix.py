from fractions import *

def display_kernel(inp: list) -> str:
    """
    Convert a list of kernel vectors (list of lists) into a formatted string.

    >>> display_kernel([[1, -2, 1]])
    '{(1, -2, 1)}'
    """
    if not inp:
        return 'empty'
    res = '{'
    for vector in inp:
        res += '('
        for item in vector:
            res += str(item) + ', '
        res = res[:-2] + ')\n'
    res = res[:-1] + '}'
    return res

def identity(n: int) -> 'Matrix':
    """
    Return the n x n identity matrix.

    >>> print(identity(3))
    | 1 0 0 |
    | 0 1 0 |
    | 0 0 1 |
    """
    return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

def updated(lst: list[list[Fraction]]) -> list:
    """
    Helper function for rref, to update row numbers during recursion.
    """
    res = []
    for item in lst:
        if item[0] == 1:
            item[1] += 1
            item[2] += 1
        elif item[0] == 2:
            item[1] += 1
        elif item[0] == 3:
            item[1] += 1
            item[2] += 1
        res.append(item)
    return res


class Matrix:
    """A class used to represent the matrix object in python

    Attributes
    - rows, list containing the rows in a matrix
    - columns, list containing the columns in a matrix
    - dimensions, tuple in form of (number of rows, number of columns)
    """
    rows: list
    columns: list
    dimensions: list

    def __init__(self, matrix: list[list[int|float|Fraction]]) -> None:
        """
        Initializes a matrix object where the matrix input is given
        as a list of rows.

        Precondition
        - Every list in matrix is the same length.
        - matrix has at least one row and at least one column

        >>> m = Matrix([[0.5, 0.25], [1.75, 2.5]])
        >>> m.rows
        [[Fraction(1, 2), Fraction(1, 4)], [Fraction(7, 4), Fraction(5, 2)]]
        >>> m.columns
        [[Fraction(1, 2), Fraction(7, 4)], [Fraction(1, 4), Fraction(5, 2)]]
        >>> m.dimensions
        [2, 2]
        """
        if not matrix or not matrix[0]:
            self.rows = []
            self.columns = []
            self.dimensions = [0, 0]
            return

        columns = []
        for i in range(len(matrix[0])):
            column = []
            for j in range(len(matrix)):
                if isinstance(matrix[j][i], Fraction):
                    column.append(matrix[j][i])
                else:
                    column.append(Fraction.from_float(matrix[j][i]).limit_denominator())
            columns.append(column)
        rows = []
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix[i])):
                if isinstance(matrix[i][j], float) or isinstance(matrix[i][j], int):
                    row.append(Fraction.from_float(matrix[i][j]).limit_denominator())
                else:
                    row.append(matrix[i][j])
            rows.append(row)
        self.rows = rows
        self.columns = columns
        self.dimensions = [len(rows), len(columns)]

    def __str__(self) -> str:
        """
        returns the string representation of this matrix

        >>> print(Matrix([[1, 2], [3, 4]]))
        | 1 2 |
        | 3 4 |
        """
        string = '| '
        for row in self.rows:
            for item in row:
                string += str(item) + ' '
            string = string[:-1] + ' |'
            string += '\n| '
        return string[:-3]

    def __eq__(self, other) -> bool:
        """
        returns true iff the two s are equal to each other.
        """
        return self.rows == other.rows and self.columns == other.columns and self.dimensions == other.dimensions

    def delete_row(self, n: int) -> list:
        """
        Deletes the nth row in a matrix

        Preconditions:
        0 <= n
        >>> m = Matrix([[1, 2], [3, 4], [5, 6]])
        >>> m.delete_row(1)
        [Fraction(3, 1), Fraction(4, 1)]
        >>> m.rows
        [[Fraction(1, 1), Fraction(2, 1)], [Fraction(5, 1), Fraction(6, 1)]]
        >>> m.columns
        [[Fraction(1, 1), Fraction(5, 1)], [Fraction(2, 1), Fraction(6, 1)]]
        >>> m.dimensions
        [2, 2]
        """
        row = self.rows.pop(n)
        for column in self.columns:
            column.pop(n)
        self.dimensions[0] -= 1
        return row

    def delete_column(self, n: int) -> list:
        """
        Deletes the nth column in a matrix

        Preconditions:
        0 <= n

        >>> m = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> m.delete_column(1)
        [Fraction(2, 1), Fraction(5, 1)]
        >>> m.rows
        [[Fraction(1, 1), Fraction(3, 1)], [Fraction(4, 1), Fraction(6, 1)]]
        >>> m.columns
        [[Fraction(1, 1), Fraction(4, 1)], [Fraction(3, 1), Fraction(6, 1)]]
        >>> m.dimensions
        [2, 2]
        """
        column = self.columns.pop(n)
        for row in self.rows:
            row.pop(n)
        self.dimensions[1] -= 1
        return column

    def determinant(self) -> Fraction|None:
        """
        return a matrix's determinant.

        Preconditions:
        - Matrix is square
        - dimensions are greater or equal to (2, 2)

        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.determinant()
        Fraction(-2, 1)

        >>> m = Matrix([[2, 5, 3], [1, -2, -1], [1, 3, 4]])
        >>> m.determinant()
        Fraction(-20, 1)
        """
        if self.dimensions[0] != self.dimensions[1]:
            return None
        if self.dimensions == [1, 1]:
            return self.rows[0][0]
        if self.dimensions == [2,2]:
            matrix = self.rows
            return matrix[0][0] * matrix[1][1] - matrix[1][0]*matrix[0][1]
        determinant = 0
        for i in range(len(self.columns)):
            new_matrix = Matrix(self.rows[1:])
            new_matrix.delete_column(i)
            determinant += self.rows[0][i] * (-1)**i *\
                           new_matrix.determinant()
        return determinant

    def transpose(self) -> None:
        """
        Turn the matrix into its transpose
        >>> m = Matrix([[5, 6, 7]])
        >>> m.transpose()
        >>> m.rows
        [[Fraction(5, 1)], [Fraction(6, 1)], [Fraction(7, 1)]]
        >>> m.columns
        [[Fraction(5, 1), Fraction(6, 1), Fraction(7, 1)]]
        >>> m.dimensions
        [3, 1]
        """
        self.rows, self.columns = self.columns, self.rows
        self.dimensions = [self.dimensions[1], self.dimensions[0]]

    def scalar_multiplication(self, n: Fraction) -> None:
        """
        Multiplies each entry in the matrix by the specified scalar

        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.scalar_multiplication(Fraction(2,1))
        >>> m.rows
        [[Fraction(2, 1), Fraction(4, 1)], [Fraction(6, 1), Fraction(8, 1)]]
        >>> m.columns
        [[Fraction(2, 1), Fraction(6, 1)], [Fraction(4, 1), Fraction(8, 1)]]
        >>> m.dimensions
        [2, 2]
        """
        new_matrix = []
        for row in self.rows:
            new_row = []
            for item in row:
                new_row.append(n*item)
            new_matrix.append(new_row)
        new = Matrix(new_matrix)
        self.rows = new.rows
        self.columns = new.columns

    def in_rref(self) -> bool:
        """
        returns True if matrix is in RREF, False otherwise

        >>> Matrix([[1, 0, 2], [0, 1, 3]]).in_rref()
        True
        >>> Matrix([[0, 1, 0], [0, 0, 1]]).in_rref()
        True
        >>> Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).in_rref()
        True
        >>> Matrix([[0, 0, 0], [0, 0, 0]]).in_rref()
        True

        >>> Matrix([[1, 2, 0], [0, 1, 3]]).in_rref()
        False
        >>> Matrix([[0, 1, 2], [0, 0, 1]]).in_rref()
        False
        >>> Matrix([[0, 1, 2], [0, 1, 0]]).in_rref()
        False
        >>> Matrix([[1, 0, 0], [1, 0, 0]]).in_rref()
        False
        >>> Matrix([[0, 0, 1], [1, 0, 0]]).in_rref()
        False
        """
        past_pivots = [-1]
        row_number = 0
        for row in self.rows:
            i = 0
            while i < len(row) and row[i] == 0 :
                i += 1
            if i == len(row):
                if row_number < len(self.rows) - 1 and self.rows[row_number + 1] != [0] * len(row):
                    return False
            else:
                if row[i] != 1:
                    return False
                if i <= max(past_pivots):
                    return False
                past_pivots.append(i)
            row_number += 1

        for index in past_pivots[1:]:
            column = self.columns[index]
            i = 0
            while i < len(column) and column[i] == 0:
                i += 1
            if i < len(column) and column[i] != 1:
                return False
        return True

    def e1(self, swap: tuple) -> None:
        """
        swaps the specified rows
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.e1((0, 1))
        >>> m.rows
        [[Fraction(3, 1), Fraction(4, 1)], [Fraction(1, 1), Fraction(2, 1)]]
        >>> m.columns
        [[Fraction(3, 1), Fraction(1, 1)], [Fraction(4, 1), Fraction(2, 1)]]
        >>> m.dimensions
        [2, 2]
        """
        self.rows[swap[0]], self.rows[swap[1]] = self.rows[swap[1]], self.rows[swap[0]]
        new_matrix = Matrix(self.rows)
        self.columns = new_matrix.columns

    def e2(self, row: int, scalar: Fraction) -> None:
        """
        multiply specified row by specified scalar

        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.e2(0, Fraction(3.0))
        >>> m.rows
        [[Fraction(3, 1), Fraction(6, 1)], [Fraction(3, 1), Fraction(4, 1)]]
        >>> m.columns
        [[Fraction(3, 1), Fraction(3, 1)], [Fraction(6, 1), Fraction(4, 1)]]
        >>> m.dimensions
        [2, 2]
        """
        lst = [scalar * item for item in self.rows[row]]
        self.rows[row] = lst
        new_matrix = Matrix(self.rows)
        self.columns = new_matrix.columns

    def e3(self, start_row: int, dest_row: int, scalar: Fraction) -> None:
        """
        perform elementary 3 operation on this matrix
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.e3(0, 1, Fraction(2))
        >>> m.rows
        [[Fraction(1, 1), Fraction(2, 1)], [Fraction(5, 1), Fraction(8, 1)]]
        >>> m.columns
        [[Fraction(1, 1), Fraction(5, 1)], [Fraction(2, 1), Fraction(8, 1)]]
        >>> m.dimensions
        [2, 2]
        """
        lst = [scalar * item for item in self.rows[start_row]]
        new_list = [self.rows[dest_row][i] + lst[i] for i in range(len(self.rows[dest_row]))]
        self.rows[dest_row] = new_list
        new_matrix = Matrix(self.rows)
        self.columns = new_matrix.columns

    def rank(self) -> int:
        """
        returns the rank of the matrix
        >>> Matrix([[1, 2], [3, 4]]).rank()
        2
        >>> Matrix([[1, 2], [2, 4]]).rank()
        1
        >>> Matrix([[0, 0], [0, 0]]).rank()
        0
        >>> Matrix([[1, 2, 3], [2, 4, 6], [1, 1, 1]]).rank()
        2
        >>> Matrix([[1, 2, 3, 4], [2, 4, 6, 8]]).rank()
        1
        """
        rref = RREF(self).rref()
        count = 0
        for row in rref.rows:
            if 1 in row:
                count += 1
        return count


    def null(self) -> int:
        """
        returns the nullity of the matrix
        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m1.null()
        0

        >>> m2 = Matrix([[1, 2, 3], [2, 4, 6]])
        >>> m2.null()
        2

        >>> m3 = Matrix([[0, 0], [0, 0]])
        >>> m3.null()
        2
        """
        return len(self.columns) - self.rank()

    def invertible(self) -> bool:
        """
        returns true iff the matrix is invertible
        >>> m1 = Matrix([[1, 2], [3, 4]])
        >>> m1.invertible()
        True

        >>> m2 = Matrix([[2, 4], [1, 2]])
        >>> m2.invertible()
        False

        >>> m3 = Matrix([[0]])
        >>> m3.invertible()
        False
        """
        return self.determinant() is not None and self.determinant() != 0

    def kernel(self) -> list:
        """
        returns list of kernel vectors

        """
        nah = RREF(self)
        nah.populate_pivots()
        rref = RREF(self).rref()
        res = []

        for i in range(len(rref.columns)):
            vector = [0] * len(rref.columns)
            column = rref.columns[i]
            if i not in nah.pivots:
                for j in range(len(column)):
                    vector[j] = column[j]
                vector[i] = -1
                res.append(vector)

        return res


class RREF:
    """
    Used to put the given matrix into Row reduced echelon form
    """
    def __init__(self, m: Matrix) -> None:
        self.matrix = m
        self.sequence = []
        self.pivots = {}

    def ref(self) -> Matrix:
        """
        Return a matrix in "almost RREF" form:
        - Each pivot is 1
        - Zeros below each pivot
        - Pivots move right
        - Values above pivots may be non-zero
        >>> m1 = Matrix([[1, 2, -1], [0, 0, 1], [0, 0, 0]])
        >>> print(RREF(m1).ref())
        | 1 2 -1 |
        | 0 0 1 |
        | 0 0 0 |

        >>> m = Matrix([[1, Fraction(1, 2)], [0, 3]])
        >>> print(RREF(m).ref())
        | 1 1/2 |
        | 0 1 |

        >>> m6 = Matrix([[2, 5, 7], [0, 4, 8], [0, 0, 6]])
        >>> print(RREF(m6).ref())
        | 1 5/2 7/2 |
        | 0 1 2 |
        | 0 0 1 |

        >>> m3 = Matrix([[1, 3, 4], [0, 1, 5], [0, 0, 1]])
        >>> print(RREF(m3).ref())
        | 1 3 4 |
        | 0 1 5 |
        | 0 0 1 |
        """
        matrix = Matrix(self.matrix.rows)

        if matrix.dimensions[0] == 1:
            for item in matrix.rows[0]:
                if item != 0:
                    self.sequence.append([2, 0, Fraction(1, item)])
                    matrix.e2(0, Fraction(1, item))
                    return matrix
            return matrix

        elif matrix.dimensions[1] == 1:
            is_it_true = False
            king_row = -1
            for i in range(len(matrix.rows)):
                if is_it_true:
                    self.sequence.append([3, king_row, i, Fraction(-matrix.rows[i][0])])
                    matrix.rows[i] = [0]
                else:
                    if matrix.rows[i] != [0]:
                        is_it_true = True
                        king_row = i
                        self.sequence.append([2, i, Fraction(1, matrix.rows[i][0])])
                        matrix.rows[i] = [1]
            return Matrix(matrix.rows)

        first = matrix.columns[0]
        indices = []
        for i in range(len(first)):
            if first[i] != 0:
                indices.append(i)

        if len(indices) != 0:
            start = indices.pop(0)

            matrix.e1((0, start))
            self.sequence.append([1, 0, start])


            self.sequence.append([2, 0, Fraction(1, matrix.rows[0][0])])
            matrix.e2(0, Fraction(1, matrix.rows[0][0]))

            for indie in indices:
                self.sequence.append([3, 0, indie, Fraction(-matrix.rows[indie][0])])
                matrix.e3(0, indie, Fraction(-matrix.rows[indie][0]))

            top_row = matrix.delete_row(0)
            top_column = matrix.delete_column(0)

            new_ref1 = RREF(matrix)
            new_ref = new_ref1.ref()
            self.sequence.extend(updated(new_ref1.sequence))
            new_ref.rows.insert(0, top_row)

            for i in range(1, len(new_ref.rows)):
                new_ref.rows[i] = [top_column[i-1]] + new_ref.rows[i]

        else:
            matrix.delete_column(0)
            new_ref1 = RREF(matrix)
            new_ref = new_ref1.ref()
            self.sequence.extend(updated(new_ref1.sequence))

            for i in range(len(new_ref.rows)):
                new_ref.rows[i] = [0] + new_ref.rows[i]

        result = Matrix(new_ref.rows)
        return result

    def rref(self) -> Matrix:
        """
        Returns a matrix in Reduced Row Echelon Form (RREF).

        >>> m1 = Matrix([[1, 2, -1], [2, 4, -2], [3, 6, -3]])
        >>> print(RREF(m1).rref())
        | 1 2 -1 |
        | 0 0 0 |
        | 0 0 0 |

        >>> m2 = Matrix([[1, 2], [3, 4]])
        >>> print(RREF(m2).rref())
        | 1 0 |
        | 0 1 |

        >>> m3 = Matrix([[0, 1, 2], [0, 0, 1]])
        >>> print(RREF(m3).rref())
        | 0 1 0 |
        | 0 0 1 |

        >>> m4 = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> print(RREF(m4).rref())
        | 1 0 -1 |
        | 0 1 2 |

        >>> m5 = Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        >>> print(RREF(m5).rref())
        | 1 0 0 |
        | 0 1 0 |
        | 0 0 1 |

        >>> m6 = Matrix([[0, 2], [0, 4]])
        >>> print(RREF(m6).rref())
        | 0 1 |
        | 0 0 |

        >>> m7 = Matrix([[2]])
        >>> print(RREF(m7).rref())
        | 1 |

        >>> m8 = Matrix([[0, 0], [0, 0]])
        >>> print(RREF(m8).rref())
        | 0 0 |
        | 0 0 |

        >>> m9 = Matrix([[2, 3], [4, 6]])
        >>> print(RREF(m9).rref())
        | 1 3/2 |
        | 0 0 |

        >>> m10 = Matrix([[1, 25/2, 18], [39/2, 27, 20], [3/2, 1/2, 13/2]])
        >>> print(RREF(m10).rref())
        | 1 0 0 |
        | 0 1 0 |
        | 0 0 1 |
        """
        ver = self.ref()
        self.populate_pivots()
        for j in self.pivots:
            column = ver.columns[j]
            to_kill = []
            for item in range(len(column)):
                if column[item] != 0:
                    to_kill.append((column[item], item))

            if to_kill:
                i = to_kill.pop()
                for this in to_kill:
                    self.sequence.append([3, i[1], this[1], Fraction(-this[0])])
                    ver.e3(i[1], this[1], Fraction(-this[0]))

        return ver

    def populate_pivots(self) -> None:
        """
        populate the pivots of matrix
        """
        rref_matrix = RREF(self.matrix).ref()
        for i in range(len(rref_matrix.rows)):
            row = rref_matrix.rows[i]
            if 1 in row:
                j = 0
                while row[j] != 1:
                    j += 1
                self.pivots[j] = i

    def inverse(self) -> Matrix:
        """
        :return: the inverse of the matrix
        precondition:
        Matrix is invertible.

        >>> m1 = Matrix([[2, 1, 1], [1, 3, 2], [1, 0, 0]])
        >>> print(RREF(m1).inverse())
        | 0 0 1 |
        | -2 1 3 |
        | 3 -1 -5 |

        >>> m1 = Matrix([[2, 1, 1], [1, 3, 2], [1, 0, 0]])
        >>> print(RREF(m1).inverse())
        | 0 0 1 |
        | -2 1 3 |
        | 3 -1 -5 |

        >>> m2 = Matrix([[4, 7], [2, 6]])
        >>> print(RREF(m2).inverse())
        | 3/5 -7/10 |
        | -1/5 2/5 |

        >>> m3 = Matrix([[1, 2], [3, 4]])
        >>> print(RREF(m3).inverse())
        | -2 1 |
        | 3/2 -1/2 |

        >>> m4 = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> print(RREF(m4).inverse())
        | 1 0 0 |
        | 0 1 0 |
        | 0 0 1 |

        >>> m5 = Matrix([[2, 0], [0, 2]])
        >>> print(RREF(m5).inverse())
        | 1/2 0 |
        | 0 1/2 |

        >>> m6 = Matrix([[1, 1], [1, 0]])
        >>> print(RREF(m6).inverse())
        | 0 1 |
        | 1 -1 |

        >>> m7 = Matrix([[3, 0], [0, 3]])
        >>> print(RREF(m7).inverse())
        | 1/3 0 |
        | 0 1/3 |

        >>> m8 = Matrix([[2, 5], [1, 3]])
        >>> print(RREF(m8).inverse())
        | 3 -5 |
        | -1 2 |
        """
        base_case = self.rref()
        for move in self.sequence:
            if move[0] == 1:
                base_case.e1((move[1], move[2]))
            elif move[0] == 2:
                base_case.e2(move[1], move[2])
            elif move[0] == 3:
                base_case.e3(move[1], move[2], move[3])
        return base_case

