from nengo.utils.cache import byte_align, bytes2human, human2bytes


def test_bytes2human():
    assert bytes2human(1) == "1.0 B"
    assert bytes2human(10000) == "9.8 KB"
    assert bytes2human(100001221) == "95.4 MB"


def test_human2bytes():
    assert human2bytes("1 MB") == 1048576
    assert human2bytes("1.5 GB") == 1610612736
    assert human2bytes("14 B") == 14
    assert human2bytes("1B") == 1
    assert human2bytes("1   B") == 1


def test_byte_align():
    assert byte_align(5, 16) == 16
    assert byte_align(23, 8) == 24
    assert byte_align(13, 1) == 13
    assert byte_align(0, 16) == 0
    assert byte_align(32, 8) == 32
