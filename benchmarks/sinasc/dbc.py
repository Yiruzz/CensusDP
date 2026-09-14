"""Reader for DATASUS `.dbc` files: PKWare DCL implode, then dBase III.

DATASUS publishes every microdata file as `.dbc`, which is a dBase III
`.dbf` whose record block is compressed with PKWare's Data Compression Library "implode"
algorithm - not zip's deflate, not anything in the standard library. There is no reader for
it installable from PyPI on Windows without a Rust toolchain: `datasus-dbc` is a maturin
package and fails at the wheel build, and `pyreaddbc` needs a C extension. The R world
solves this with `read.dbc`, which wraps the same C code this module reimplements.

So the decompressor is written out here, in pure Python and with no dependencies, as a
straight translation of `contrib/blast/blast.c` from zlib (Mark Adler's public-domain
implementation of the format). It is ~120 lines and it is deterministic - there is no
heuristic anywhere in it - so the risk of carrying our own copy is low and the alternative
is a build toolchain in the way of `python -m benchmarks.sinasc.prepare`.

THE FILE LAYOUT, which is the only part specific to DATASUS rather than to PKWare:

    bytes 0..header_size      the dBase III header, STORED UNCOMPRESSED
    bytes 8..10               header_size, uint16 little-endian
    bytes header_size..+4     4 bytes, skipped
    the rest                  the record block, implode-compressed

so `dbc_to_dbf` copies the header verbatim, skips the four bytes and blasts the remainder.

Performance note: `_Blast.run` copies matched bytes one at a time because a match may
overlap its own output (distance < length), which is how run-length encoding falls out of
LZ77 and which a slice copy would get wrong. That costs ~40 s for a 25 MB `.dbc`. It only
ever runs once per file, in prepare_data.py, and the result is cached as Parquet.
"""

import struct
from typing import Dict, List, Tuple

MAXBITS = 13

# The three Huffman tables of the implode format, run-length encoded: in each byte the high
# nibble plus one is a repeat count and the low nibble is a code length. Verbatim from
# blast.c.
_LITLEN = bytes([
    11, 124, 8, 7, 28, 7, 188, 13, 76, 4, 10, 8, 12, 10, 12, 10, 8, 23, 8,
    9, 7, 6, 7, 8, 7, 6, 55, 8, 23, 24, 12, 11, 7, 9, 11, 12, 6, 7, 22, 5,
    7, 24, 6, 11, 9, 6, 7, 22, 7, 11, 38, 7, 9, 8, 25, 11, 8, 11, 9, 12,
    8, 12, 5, 38, 5, 38, 5, 11, 7, 5, 6, 21, 6, 10, 53, 8, 7, 24, 10, 27,
    44, 253, 253, 253, 252, 252, 252, 13, 12, 45, 12, 45, 12, 61, 12, 45,
    44, 173])
_LENLEN = bytes([2, 35, 36, 53, 38, 23])
_DISTLEN = bytes([2, 20, 53, 230, 247, 151, 248])

# Length codes: base length per symbol, and how many extra bits follow.
_BASE = [3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 40, 72, 136, 264]
_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

# The length that means "end of stream" rather than a real match.
_END_OF_STREAM = 519


class _Huffman:
    """A canonical Huffman table, decoded from blast.c's run-length format.

    Attributes:
        count: count[n] is how many codes have length n.
        symbol: symbols ordered canonically, so index (offset + code - first) resolves one.
    """

    def __init__(self, packed: bytes):
        lengths: List[int] = []
        for byte in packed:
            lengths.extend([byte & 15] * ((byte >> 4) + 1))

        self.count = [0] * (MAXBITS + 1)
        for length in lengths:
            self.count[length] += 1

        offsets = [0] * (MAXBITS + 2)
        for length in range(1, MAXBITS + 1):
            offsets[length + 1] = offsets[length] + self.count[length]

        self.symbol = [0] * len(lengths)
        for sym, length in enumerate(lengths):
            if length:
                self.symbol[offsets[length]] = sym
                offsets[length] += 1


class _Blast:
    """PKWare DCL implode decompressor. One instance per stream."""

    def __init__(self, data: bytes):
        self.inp = data
        self.incnt = 0
        self.bitbuf = 0
        self.bitcnt = 0
        self.out = bytearray()

    def bits(self, need: int) -> int:
        """Read `need` bits, least-significant first."""
        val = self.bitbuf
        while self.bitcnt < need:
            val |= self.inp[self.incnt] << self.bitcnt
            self.incnt += 1
            self.bitcnt += 8
        self.bitbuf = val >> need
        self.bitcnt -= need
        return val & ((1 << need) - 1)

    def decode(self, table: _Huffman) -> int:
        """Decode one symbol. The codes are stored inverted, hence the `^ 1`."""
        code = first = index = 0
        length = 1
        bitbuf = self.bitbuf
        left = self.bitcnt
        nxt = 1
        while True:
            while left:
                left -= 1
                code |= (bitbuf & 1) ^ 1
                bitbuf >>= 1
                count = table.count[nxt]
                nxt += 1
                if code < first + count:
                    self.bitbuf = bitbuf
                    self.bitcnt = (self.bitcnt - length) & 7
                    return table.symbol[index + (code - first)]
                index += count
                first = (first + count) << 1
                code <<= 1
                length += 1
            left = (MAXBITS + 1) - length
            if left == 0:
                raise ValueError('incomplete Huffman code in the .dbc stream')
            bitbuf = self.inp[self.incnt]
            self.incnt += 1
            left = min(left, 8)

    def run(self) -> bytes:
        """Decompress the whole stream.

        Returns:
            bytes: The uncompressed record block.

        Raises:
            ValueError: On a malformed header or an out-of-range back-reference.
        """
        litcode = _Huffman(_LITLEN)
        lencode = _Huffman(_LENLEN)
        distcode = _Huffman(_DISTLEN)

        coded_literals = self.bits(8)
        if coded_literals > 1:
            raise ValueError(f'bad literal flag in .dbc header: {coded_literals}')
        dict_bits = self.bits(8)
        if not 4 <= dict_bits <= 6:
            raise ValueError(f'bad dictionary size in .dbc header: {dict_bits}')

        out = self.out
        while True:
            if self.bits(1):
                symbol = self.decode(lencode)
                length = _BASE[symbol] + self.bits(_EXTRA[symbol])
                if length == _END_OF_STREAM:
                    break
                nbits = 2 if length == 2 else dict_bits
                distance = (self.decode(distcode) << nbits) + self.bits(nbits) + 1
                if distance > len(out):
                    raise ValueError('back-reference before the start of the .dbc stream')
                start = len(out) - distance
                # Byte at a time on purpose: a match may overlap its own output.
                for i in range(length):
                    out.append(out[start + i])
            else:
                out.append(self.bits(8) if not coded_literals else self.decode(litcode))
        return bytes(out)


def dbc_to_dbf(path) -> bytes:
    """Decompress a DATASUS `.dbc` into the bytes of the equivalent `.dbf`.

    Args:
        path: The `.dbc` file.

    Returns:
        bytes: A well-formed dBase III file.
    """
    raw = open(path, 'rb').read()
    header_size = struct.unpack('<H', raw[8:10])[0]
    return raw[:header_size] + _Blast(raw[header_size + 4:]).run()


def read_dbf(data: bytes, encoding: str = 'latin-1') -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Parse a dBase III file into column names and rows of stripped strings.

    EVERYTHING COMES BACK AS str, deliberately. DATASUS codes are zero-padded ('01', '12',
    '350000') and several carry a non-numeric sentinel, so typing them would either lose the
    padding or fail outright - the same reason the French and Spanish domains are strings.
    See sinasc_domains.py.

    Args:
        data: The bytes of a `.dbf`, e.g. from dbc_to_dbf.
        encoding: Character encoding of the fields. DATASUS writes latin-1.

    Returns:
        Tuple[List[str], List[Tuple[str, ...]]]: Field names, then one tuple per live record.
    """
    n_records = struct.unpack('<I', data[4:8])[0]
    header_size = struct.unpack('<H', data[8:10])[0]
    record_size = struct.unpack('<H', data[10:12])[0]

    fields: List[Tuple[str, int]] = []
    offset = 32
    while data[offset] != 0x0D:            # 0x0D terminates the field descriptor array
        name = data[offset:offset + 11].split(b'\x00')[0].decode('ascii')
        fields.append((name, data[offset + 16]))
        offset += 32

    names = [name for name, _ in fields]
    rows = []
    position = header_size
    for _ in range(n_records):
        record = data[position:position + record_size]
        position += record_size
        if len(record) < record_size or record[:1] == b'*':     # deleted
            continue
        values = []
        cursor = 1                          # byte 0 is the deletion flag
        for _, width in fields:
            values.append(record[cursor:cursor + width].decode(encoding, 'replace').strip())
            cursor += width
        rows.append(tuple(values))
    return names, rows


def read_dbc(path, encoding: str = 'latin-1') -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Decompress and parse a `.dbc` in one call.

    Args:
        path: The `.dbc` file.
        encoding: Character encoding of the fields.

    Returns:
        Tuple[List[str], List[Tuple[str, ...]]]: Field names, then rows.
    """
    return read_dbf(dbc_to_dbf(path), encoding=encoding)


def field_widths(path) -> Dict[str, int]:
    """Field name -> declared width, without materialising the records.

    Useful for checking a year's layout before committing to a download of all 27 states.

    Args:
        path: The `.dbc` file.

    Returns:
        Dict[str, int]: Declared width per field, in file order.
    """
    data = dbc_to_dbf(path)
    widths: Dict[str, int] = {}
    offset = 32
    while data[offset] != 0x0D:
        name = data[offset:offset + 11].split(b'\x00')[0].decode('ascii')
        widths[name] = data[offset + 16]
        offset += 32
    return widths
