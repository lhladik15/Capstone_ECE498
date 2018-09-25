__author__ = "Adam Jarzebak"
__copyright__ = "Copyright 2018, Adam Jarzebak"
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Adam Jarzebak"
__email__ = "adam@jarzebak.eu"
import random
import string


def generate_random_string(size: int=16) -> str:
    """
    Generates a random string with 16 characters. Size value can be changed.
    :return: random string: str
    """
    random_string = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(size)])
    return random_string


def generate_random_number(size: int=8) -> str:
    """
    Generates a random number string for given size.
    :param size:
    :return:
    """
    x = [random.randint(0, 9) for p in range(0, size)]
    return ''.join(str(a) for a in x)


def grid_reference_to_northing_easting(grid_reference):
    """
    Needs to include reference
    :param grid_reference:
    :return:
    """
    grid_reference = grid_reference.strip().replace(' ', '')
    if len(grid_reference) == 0 or len(grid_reference) % 2 == 1 or len(grid_reference) > 12:
        return None, None
    grid_reference = grid_reference.upper()
    if grid_reference[0] not in 'STNOH' or grid_reference[1] == 'I':
        return None, None
    e = n = 0
    c = grid_reference[0]
    if c == 'T':
        e = 500000
    elif c == 'N':
        n = 500000
    elif c == 'O':
        e = 500000
        n = 500000
    elif c == 'H':
        n = 1000000
    c = ord(grid_reference[1]) - 66
    if c < 8:  # J
        c += 1
    e += (c % 5) * 100000
    n += (4 - c/5) * 100000
    c = grid_reference[2:]
    try:
        s = c[:int(len(c)/2)]
        while len(s) < 5:
            s += '0'
        e += int(s)
        s = c[int(-len(c)/2):]
        while len(s) < 5:
            s += '0'
        n += int(s)
    except Exception as error:
        print("Caught exception during conversion. Issue: {}".format(error))
        return None, None
    # Data is converted into integers
    return int(e), int(n)


if __name__ == '__main__':
    grid_reference_to_northing_easting('TQ1026048120')
