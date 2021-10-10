import pytest
from data_processing import DataProcessing
from file_read import FileRead


@pytest.fixture
def init():
    path = 'testdata/01_14T_01.dat'
    return FileRead(path).to_dataframe()


class TestGroup:

    def test_interpolate(self, init):



