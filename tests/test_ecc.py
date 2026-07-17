"""ECC encode/decode tests."""

from vsteg import ecc


def test_ecc_roundtrip():
    data = b"hello reed solomon" * 50
    encoded = ecc.encode(data)
    assert len(encoded) > len(data)
    assert ecc.decode(encoded) == data


def test_ecc_survives_bit_flips():
    data = b"abcdef" * 100
    encoded = bytearray(ecc.encode(data))
    # Flip a few bytes spread out (interleaving helps)
    for i in (10, 100, 250, 500):
        if i < len(encoded):
            encoded[i] ^= 0xFF
    assert ecc.decode(bytes(encoded)) == data
