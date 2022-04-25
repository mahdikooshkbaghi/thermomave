from thermomave import __version__
from thermomave import utils


def test_version():
    assert __version__ == '0.1.2'


def test_load_dataset():
    """
    Test that utils.get_data() returns correct L and C.
    """
    filename = './data/gb1.csv.gz'
    res = utils.load_dataset(filename)
    assert res[2] == 55
    assert res[3] == 20
