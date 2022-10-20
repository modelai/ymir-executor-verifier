from pprint import pprint


def print_error(result):
    for key in result:
        if result.get('error'):
            pprint({key: result[key]})
