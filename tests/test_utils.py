# pytest library
import pytest

# import climatePy utils module
import climatePy._utils as climatePy

# standard python libs
import re
from datetime import datetime

# old imports
# from climatePy import utils
# from src.climatePy import utils

def test_getExtension():
    # Test cases where the input string contains a file extension
    assert climatePy.getExtension('myfile.txt') == 'txt'
    assert climatePy.getExtension('my.file.with.multiple.dots.txt') == 'txt'
    assert climatePy.getExtension('/path/to/myfile.txt') == 'txt'
    assert climatePy.getExtension('/path/to/my.file.with.multiple.dots.txt') == 'txt'

    # Test cases where the input string does not contain a file extension
    assert climatePy.getExtension('myfile') == ''
    assert climatePy.getExtension('mypath/myfile') == ''
    assert climatePy.getExtension('mypath/myfile.') == ''
    assert climatePy.getExtension('mypath/.myfile') == 'myfile'
    assert climatePy.getExtension('') == ''

def test_format_units():
    # Test 1: No interval or unit string provided
    assert climatePy.format_units() is None

    # Test 2: Interval and unit strings are the same
    assert climatePy.format_units("3 hours", "hours") == "3 hours"

    assert climatePy.format_units("3 hours hours", "hours") == "3 hours"

    # Test 3: Interval and unit strings are different
    assert climatePy.format_units("2 weeks", "days") == "2 weeks"

    assert climatePy.format_units("2 weeks weeks", "days") == "2 weeks weeks"

    assert climatePy.format_units("2 weeks weeks", "weeks") == "2 weeks"
    
    # Test 4: Interval string contains multiple occurrences of unit string
    assert climatePy.format_units("5 days days days", "days") == "5 days"

    assert climatePy.format_units("5 days days days", "day") == '5 days days days'
    
    # Test 5: Interval string is a single word
    assert climatePy.format_units("seconds", "seconds") == "seconds"

    # Test 6: Interval string is a combination of words not containing unit string
    assert climatePy.format_units("6 minutes and 30 seconds", "hours") == "6 minutes and 30 seconds"

    assert climatePy.format_units("6 minutes and 30 seconds", "6 minutes and 30 seconds") == "6 minutes and 30 seconds"

def test_format_date():
    # Test case 1
    date_str = '2022-03-12 09:30:00'
    expected_output = '2022-03-12'
    assert climatePy.format_date(date_str) == expected_output

    # Test case 2
    date_str = '2022-03-12T09:30:00'
    expected_output = '2022-03-12'
    assert climatePy.format_date(date_str) == expected_output

    # Test case 3
    date_str = '2022-03-12T09:30:00Z'
    expected_output = '2022-03-12'
    assert climatePy.format_date(date_str) == expected_output

    # Test case 4
    date_str = '2022-03-12T09:30:00-08:00'
    expected_output = '2022-03-12'
    assert climatePy.format_date(date_str) == expected_output

    # Test case 5
    date_str = '2022-03-12'
    expected_output = '2022-03-12'
    assert climatePy.format_date(date_str) == expected_output

    # Test case 6
    date_str = '2022-03-1' # Invalid date format
    expected_output = ''
    assert climatePy.format_date(date_str) == expected_output
