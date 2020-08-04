from nengo import rc


def test_read_file(tmp_path):
    filepath = str(tmp_path / "test_rc_file.txt")

    # write the test file
    with open(filepath, "w") as fh:
        fh.write("[test_rc_section]\nsetting0 = 3\nsetting1=5")

    del fh

    # read the test file using `rc.read_file`
    with open(filepath, "r") as fh:
        rc.read_file(fh)

    assert int(rc["test_rc_section"]["setting0"]) == 3
    assert int(rc["test_rc_section"]["setting1"]) == 5
