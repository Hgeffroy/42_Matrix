class Vector:
    """
    Class representing a vector of int.
    """

    def __init__(self, coordinates: list):
        if all(type(n) is int for n in coordinates) is False:
            raise TypeError("All coordinates must be integers")

        self._size = len(coordinates)
        self._coordinates = coordinates

    def __str__(self):
        string = f'Vector size: {self._size}\n'f'Coordinates:\n'
        for coord in self._coordinates:
            string += f'{coord}'

        return string


