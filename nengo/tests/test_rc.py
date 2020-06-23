import os
from nengo import rc


def test_read_file():
    """Tests if files are read properly"""

    file_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )

    fp = open("%s/nengo/tests/test_rc_file1.txt" % file_dir, "r")

    fp2 = open("%s/nengo/tests/test_rc_file1.txt" % file_dir, "r")

    class TestFile:
        """This class is iterable"""

        # name = None
        source = "source"
        mode = "r"
        test_list = ["['header']", "line2:value2", "line3:value3"]
        # encoding = "cp1252"

        def __iter__(self):
            """Returns the Iterator object"""
            return TestFileIterator(self)

    class TestFileIterator:
        """Iterator class"""

        def __init__(self, contents):
            self._contents = contents.test_list
            self._index = 0

        def __next__(self):
            """Returns next value"""
            if self._index < len(self._contents):
                result = self._contents[self._index]
                self._index += 1
                return result
            raise StopIteration

    fp3 = TestFile()

    rc.read_file(fp, filename="test_rc_file1.txt")
    rc.read_file(fp2, filename=None)

    fp.close()
    fp2.close()

    rc.read_file(fp3, filename=None)
