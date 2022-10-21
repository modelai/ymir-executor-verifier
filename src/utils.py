from pprint import pprint
from typing import Dict


def print_error(result: Dict):
    error_count = 0
    for key in result:
        if result.get('error'):
            error_count += 1
            pprint({key: result[key]})

    if error_count == 0:
        print('nice, no error found')
