from vsteg.sniff import guess_extension, suggested_filename


def test_text():
    assert guess_extension(b"hello secret world\n") == ".txt"


def test_png():
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    assert guess_extension(data) == ".png"


def test_pdf():
    assert guess_extension(b"%PDF-1.4\n%") == ".pdf"


def test_json():
    assert guess_extension(b'{"a": 1}') == ".json"


def test_unknown():
    assert guess_extension(b"\x00\x01\x02\xff") == ".bin"


def test_suggested_name():
    assert suggested_filename(b"plain text") == "recovered.txt"
