import os

import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo.utils.nco as nco
from nengo.utils.nco import Subfile


@pytest.fixture
def data():
    return '0123456789\n123456789'


@pytest.fixture
def testfile(data, tmpdir):
    f = tmpdir.join('file.txt')
    f.write(data)
    return f


class TestSubfile(object):
    def test_reads_only_from_start_to_end(self, data, testfile):
        with testfile.open() as f:
            assert Subfile(f, 2, 6).read() == data[2:6]
        with testfile.open() as f:
            assert Subfile(f, 2, 6).read(8) == data[2:6]
        with testfile.open() as f:
            assert Subfile(f, 2, 6).readline() == data[2:6]
        with testfile.open() as f:
            assert Subfile(f, 2, 6).readline(8) == data[2:6]

    def test_respects_read_size(self, data, testfile):
        with testfile.open() as f:
            assert Subfile(f, 2, 6).read(2) == data[2:4]
        with testfile.open() as f:
            assert Subfile(f, 2, 6).readline(2) == data[2:4]

    def test_readline(self, data, testfile):
        with testfile.open() as f:
            assert Subfile(f, 2, 14).readline() == data[2:11]
        with testfile.open() as f:
            assert Subfile(f, 2, 14).readline(15) == data[2:11]

    def test_seek(self, data, testfile):
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.seek(2)
            assert sf.read() == data[4:6]
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.read(1)
            sf.seek(2, os.SEEK_CUR)
            assert sf.read() == data[5:6]
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.seek(-2, os.SEEK_END)
            assert sf.read() == data[4:6]

    def test_seek_before_start(self, data, testfile):
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.seek(-2)
            assert sf.read() == data[2:6]
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.read(1)
            sf.seek(-4, os.SEEK_CUR)
            assert sf.read() == data[2:6]
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.seek(-8, os.SEEK_END)
            assert sf.read() == data[2:6]

    def test_seek_after_end(self, testfile):
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.seek(8)
            assert sf.read() == ''
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.read(1)
            sf.seek(8, os.SEEK_CUR)
            assert sf.read() == ''
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.seek(8, os.SEEK_END)
            assert sf.read() == ''


def test_nco_roundtrip(tmpdir):
    tmpfile = tmpdir.join('test.nco')

    pickle_data = {'0': 237, 'str': 'foobar'}
    array = np.array([[4, 3], [2, 1]])

    with tmpfile.open('wb') as f:
        nco.write(f, pickle_data, array)

    with tmpfile.open('rb') as f:
        pickle_data2, array2 = nco.read(f)

    assert pickle_data == pickle_data2
    assert_equal(array, array2)
