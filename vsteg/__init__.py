"""vsteg — video steganography toolkit (encode, decode, detect)."""

__version__ = "0.1.0"

METHOD_APPEND = 0
METHOD_LSB = 1
METHOD_DCT = 2

METHOD_NAMES = {
    METHOD_APPEND: "append",
    METHOD_LSB: "lsb",
    METHOD_DCT: "dct",
}

FLAG_ENCRYPTED = 0x01
FLAG_COMPRESSED = 0x02
FLAG_ECC = 0x04

MAGIC = b"VSTG"
HEADER_SIZE = 47  # 4+1+1+1+16+12+8+4
