class Matrix:
    """
    Class representing a matrix of int.
    """

    def __init__(self, values: list):
        for v in values:
            if all(type(n) is int for n in v) is False:
                raise TypeError("All values must be integers")

        self._size = len(values)
        self._values = values

    def __str__(self):
        string = f'Vector size: {self._size}\n'f'Coordinates:\n'
        for coord in self._values:
            string += f'{coord}'

        return string


