__author__ = "Adam Jarzebak"
__copyright__ = "Copyright 2018, Adam Jarzebak"
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Adam Jarzebak"
__email__ = "adam@jarzebak.eu"
"""
Top level function helping with reading different types of files. Counting, matching and other helpers regards to data science.
"""
import csv


def read_tsv_file(filename: str) -> list:
    """
    This function is opening a file and reading a data into a list object.
    :type filename: str
    return: list of rows from tsv file: list
    """
    data_from_file = []
    with open(filename, 'r') as opened_file:
        reader = csv.reader(opened_file, delimiter='\t')
        for i, row in enumerate(reader):
            data_from_file.append(row)
    return data_from_file


def display_column_from_tsv_file(data: list, column: int) -> None:
    """
    Display field indicated by column number, for each of the rows
    :param data:
    :param column:
    :return:
    """
    assert 0 > column < int, "No such column, please ensure that index is correct"
    for row in data:
        try:
            print(row[column])
        except IndexError:
            print("No such column")


def check_for_empty_field_in_column(data: list, column: int, print_all_row: bool = False) -> None:
    """
    Function is retrieving last date when update was performed
    :return: last update info : str
    """
    total_empty_fields = 0
    for row_index, row in enumerate(data):
        try:
            if row[column] == "":
                if print_all_row:
                    print("Empty row in index: {}.\nContent: {}".format(row_index, row))
                else:
                    total_empty_fields += 1
        except IndexError:
            print("No such column")
    print("Summary: {} empty fields in dataset. Total length: {}".format(
        total_number_fields, len(data)))


def count_unique_value_in_column(data: list, column: int) -> int:
    """
    Count unique number of items in column. If column number is incorrect returns -1
    :param data:
    :param column:
    :return: Number of unique fields or -1 if column number is incorrect
    """
    unique_values = set()
    if column < 0 and column <= len(data):
        return -1
    for row_index, row in enumerate(data):
        unique_values.add(row[column])
    return len(unique_values)


payload = {'_ctl4%3A_ctl1%3AgrpCountry:': 'radLocatedIn', '_ctl4%3A_ctl1%3AchkAll': 'on',
           '_ctl4:_ctl1:chkAll': 'on', '_ctl4:_ctl1:chkEngland': 'on',
           '_ctl4:_ctl1:chkScotland': 'on',
           '_ctl4:_ctl1:chkWales': 'on',
           '_ctl4:_ctl1:chkNorthernIreland': 'on',
           '_ctl4:_ctl1:chkOther': 'on',
           '_ctl4:_ctl1:dropCounty': '0',
           '_ctl4:_ctl1:grdScotland:_ctl5:RAVE0001.x': '19',
           '_ctl4:_ctl1:grdScotland:_ctl5:RAVE0001.y': '14',
           '_ctl4:_ctl1:popView:ValueReturned': 'false',
           '_ctl4:_ctl1:popView:HiddenModal': 'True',
           '_ctl4:_ctl1:popView:HiddenL': '500',
           '_ctl4:_ctl1:popView:HiddenT': '500',
           '_ctl4:_ctl1:popView:HiddenH': '700',
           '_ctl4:_ctl1:popView:HiddenW': '600',
           '_ctl4:_ctl1:popView:HiddenCenter': 'True',
           '_ctl4:_ctl1:popView:HiddenSizeable': 'no',
           '_ctl4:_ctl1:popView:HiddenPopupName': 'Register Details',
           '_ctl4:_ctl1:popView:HiddenAllowScrollbars': 'no'
           }
