import os
import struct

import numpy as np
import pytest
from numpy.testing import assert_equal

import nengo.utils.nco as nco
from nengo.exceptions import CacheIOError
from nengo.utils.nco import Subfile


@pytest.fixture(name="data")
def fixture_data():
    return "0123456789\n123456789"


@pytest.fixture(name="testfile")
def fixture_testfile(data, tmpdir):
    f = tmpdir.join("file.txt")
    f.write(data)
    return f


def write_custom_nco_header(fileobj, magic_string="NCO", version=0):
    magic_string = magic_string.encode("utf-8")
    header_format = "@{}sBLLLL".format(len(magic_string))
    assert struct.calcsize(header_format) == nco.HEADER_SIZE

    header = struct.pack(header_format, magic_string, version, 0, 1, 2, 3)
    fileobj.write(header)


class TestSubfile:
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

    def test_readinto(self, data, testfile):
        b = bytearray(4)
        with testfile.open("rb") as f:
            assert Subfile(f, 2, 6).readinto(b) == 4
        assert b == b"2345"

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
            assert sf.read() == ""
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.read(1)
            sf.seek(8, os.SEEK_CUR)
            assert sf.read() == ""
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            sf.seek(8, os.SEEK_END)
            assert sf.read() == ""

    def test_tell(self, data, testfile):
        with testfile.open() as f:
            sf = Subfile(f, 2, 6)
            assert sf.tell() == 0
            sf.seek(3)
            assert sf.tell() == 3

    def test_read_errors(self, tmpdir):
        # use a bad magic string
        filepath = str(tmpdir.join("bad_magic_cache_file.txt"))
        with open(filepath, "wb") as fh:
            write_custom_nco_header(fh, magic_string="BAD")

        with open(filepath, "rb") as fh:
            with pytest.raises(CacheIOError, match="Not a Nengo cache object file"):
                nco.read(fh)

        # use a bad version number
        filepath = str(tmpdir.join("bad_version_cache_file.txt"))
        with open(filepath, "wb") as fh:
            write_custom_nco_header(fh, version=255)

        with open(filepath, "rb") as fh:
            with pytest.raises(CacheIOError, match="NCO protocol version 255 is"):
                nco.read(fh)


def test_nco_roundtrip(tmpdir):
    tmpfile = tmpdir.join("test.nco")

    pickle_data = {"0": 237, "str": "foobar"}
    array = np.array([[4, 3], [2, 1]])

    with tmpfile.open("wb") as f:
        nco.write(f, pickle_data, array)

    with tmpfile.open("rb") as f:
        pickle_data2, array2 = nco.read(f)

    assert pickle_data == pickle_data2
    assert_equal(array, array2)
