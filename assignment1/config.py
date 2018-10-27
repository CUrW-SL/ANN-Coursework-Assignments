DATA_SET_FP = '/home/nira/PycharmProjects/Neural-Networks/assignment1/data-set/crx.data'

attribute_formation = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
                       'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']

attribute_types = {
    'A1': ['b', 'a'],
    'A2': 'float',
    'A3': 'float',
    'A4': ['u', 'y', 'l', 't'],
    'A5': ['g', 'p', 'gg'],
    'A6': ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'],
    'A7': ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'],
    'A8': 'float',
    'A9': ['t', 'f'],
    'A10': ['t', 'f'],
    'A11': 'int',
    'A12': ['t', 'f'],
    'A13': ['g', 'p', 's'],
    'A14': 'str',
    'A15': 'int',
    'A16': ['+', '-']
}


def target_converter(target):
    return 1 if target == '+' else 0


attribute_converters = {'A1': lambda x: x,
                        'A2': float,
                        'A3': float,
                        'A4': lambda x: x,
                        'A5': lambda x: x,
                        'A6': lambda x: x,
                        'A7': lambda x: x,
                        'A8': float,
                        'A9': lambda x: x,
                        'A10': lambda x: x,
                        'A11': int,
                        'A12': lambda x: x,
                        'A13': lambda x: x,
                        'A14': lambda x: x,
                        'A15': int,
                        'A16': lambda target: 1 if target == '+' else 0}
