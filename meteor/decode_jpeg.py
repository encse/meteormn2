# lrpt_msumr_block_decode.py
# Decode a single 8x8 block from LRPT/MSU-MR "JPEG-like" entropy payload,
# matching the logic of the first C++ snippet (Huffman -> dequant -> IDCT -> pixels).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple



STANDARD_QUANTIZATION_TABLE: List[int] = [
    16, 11, 10, 16, 24, 40, 51, 61, 
    12, 12, 14, 19, 26, 58, 60, 55, 
    14, 13, 16, 24, 40, 57, 69, 56, 
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 
    24, 35, 55, 64, 81, 104, 113, 92, 
    49, 64, 78, 87, 103, 121, 120, 101, 
    72, 92, 95, 98, 112, 100, 103, 99
]

ZIGZAG: List[int] = [
   0, 1, 5, 6, 14, 15, 27, 28, 
   2, 4, 7, 13, 16, 26, 29, 42, 
   3, 8, 12, 17, 25, 30, 41, 43, 
   9, 11, 18, 24, 31, 40, 44, 53, 
   10, 19, 23, 32, 39, 45, 52, 54, 
   20, 22, 33, 38, 46, 51, 55, 60, 
   21, 34, 37, 47, 50, 56, 59, 61, 
   35, 36, 48, 49, 57, 58, 62, 63
]

T_AC_0: List[int] = [
    0, 2, 1, 3, 3, 2, 4, 3, 
    5, 5, 4, 4, 0, 0, 1, 125, 
    1, 2, 3, 0, 4, 17, 5, 18, 
    33, 49, 65, 6, 19, 81, 97, 7, 
    34, 113, 20, 50,129, 145, 161, 8, 
    35, 66, 177, 193, 21, 82, 209, 240, 
    36, 51, 98, 114, 130, 9, 10, 22, 
    23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 
    72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 
    103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 
    135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 
    165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 
    194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 
    216, 217, 218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 
    243, 244, 245, 246, 247, 248, 249, 250
]

DC_CAT_OFF: List[int] = [2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9]


class BitIOConst:
    def __init__(self, data: bytes):
        self._data = data
        self._size = len(data)
        self._pos = 0  # bit position

    def peek_bits(self, n: int) -> int:
        result = 0
        bit = 0

        for i in range(n):
            p = self._pos + i
            byte_index = p >> 3

            if byte_index < self._size: 
                bit = (self._data[byte_index] >> (7 - (p & 7))) & 1
            else:
                bit = 0

            result = (result << 1) | bit

        return result

    def advance_bits(self, n: int) -> None:
        self._pos += n

    def fetch_bits(self, n: int) -> int:
        result = self.peek_bits(n)
        self.advance_bits(n)
        return result

@dataclass
class AcEntry:
    run: int
    size: int
    length: int  # huffman code length
    mask: int
    code: int
    
def fill_dqt_by_q(q: float) -> List[int]:
    assert q >= 0
    dqt = [0] * 64

    if 20 < q < 50:
        f = 5000.0 / q
    else:
        f = 200.0 - 2.0 * q

    for i in range(64):
        dqt[i] = int(math.floor((f / 100.0) * STANDARD_QUANTIZATION_TABLE[i] + 0.5))
        if dqt[i] < 1:
            dqt[i] = 1

    return dqt


def map_range(cat: int, vl: int) -> int:
    if cat == 0:
        return 0
    assert cat >= 1
    
    maxval = (1 << cat) - 1
    sig = (vl >> (cat - 1)) != 0
    return vl if sig else vl - maxval


def get_dc_real(word: int) -> int:
    assert 0 <= word <= 0xFFFF

    match word >> 14:
        case 0:
            return 0
        case _:
            match word >> 13:
                case 2:
                    return 1
                case 3:
                    return 2
                case 4:
                    return 3
                case 5:
                    return 4
                case 6:
                    return 5
                case _:
                    if (word >> 12) == 0x00E:
                        return 6
                    if (word >> 11) == 0x01E:
                        return 7
                    if (word >> 10) == 0x03E:
                        return 8
                    if (word >> 9) == 0x07E:
                        return 9
                    if (word >> 8) == 0x0FE:
                        return 10
                    if (word >> 7) == 0x1FE:
                        return 11
    return -1


def init_huffman_table():
    v = [0] * 65536

    min_code = [0] * 17
    maj_code = [0] * 17

    p = 16
    for k in range(1, 17):
        for i in range(T_AC_0[k - 1]):
            v[(k << 8) + i] = T_AC_0[p]
            p += 1

    code = 0
    for k in range(1, 17):
        min_code[k] = code
        code += T_AC_0[k - 1]
        maj_code[k] = code - (1 if code != 0 else 0)
        code *= 2

        if T_AC_0[k - 1] == 0:
            min_code[k] = 0xFFFF
            maj_code[k] = 0

    ac_table: List[AcEntry] = []

    n = 0
    for k in range(1, 17):
        min_val = min_code[k]
        max_val = maj_code[k]

        for i in range(1 << k):
            if i <= max_val and i >= min_val:
                size_val = v[(k << 8) + i - min_val]
                run = size_val >> 4
                size = size_val & 0xF

                ac_table.append(
                    AcEntry(
                        run=run,
                        size=size,
                        length=k,
                        mask=(1 << k) - 1,
                        code=i,
                    )
                )
                n += 1

    ac_lookup = [0] * 65536
    for i in range(65536):
        ac_lookup[i] = get_ac_real(i, ac_table=ac_table)

    dc_lookup = [0] * 65536
    for i in range(65536):
        dc_lookup[i] = get_dc_real(i)

    return ac_table, ac_lookup, dc_lookup


def get_ac_real(word: int, ac_table: List[AcEntry]) -> int:
    assert 0 <= word <= 0xFFFF

    for i in range(162):
        if (((word >> (16 - ac_table[i].length)) & ac_table[i].mask) == ac_table[i].code):
            return i

    return -1


def init_cos():
    mCosine = [[0.0 for _ in range(8)] for _ in range(8)]
    mAlpha = [0.0 for _ in range(8)]

    for y in range(8):
        for x in range(8):
            mCosine[y][x] = math.cos(math.pi / 16.0 * (2 * y + 1) * x)

    for x in range(8):
        mAlpha[x] = (1.0 / math.sqrt(2.0)) if (x == 0) else 1.0

    return mCosine, mAlpha

class Image:
    def __init__(self) -> None:
        self._ac_table, self._ac_lookup, self._dc_lookup = init_huffman_table()
        self._cosine, self._alpha = init_cos()

    def filt_idct8x8(self, inp: List[float]) -> None:
        assert len(inp) == 64

        res = [0] * 64
        for y in range(8):
            for x in range(8):
                s = 0.0
                for u in range(8):
                    cxu = self._alpha[u] * self._cosine[x][u]
                    s += cxu * (
                        inp[0 * 8 + u] * self._alpha[0] * self._cosine[y][0] +
                        inp[1 * 8 + u] * self._alpha[1] * self._cosine[y][1] +
                        inp[2 * 8 + u] * self._alpha[2] * self._cosine[y][2] +
                        inp[3 * 8 + u] * self._alpha[3] * self._cosine[y][3] +
                        inp[4 * 8 + u] * self._alpha[4] * self._cosine[y][4] +
                        inp[5 * 8 + u] * self._alpha[5] * self._cosine[y][5] +
                        inp[6 * 8 + u] * self._alpha[6] * self._cosine[y][6] +
                        inp[7 * 8 + u] * self._alpha[7] * self._cosine[y][7]
                    )
                res[y * 8 + x] = s / 4.0
        return res

    def decode_14_blocks(
        self,
        payload: bytes,
        qf: float,
    ) -> list[list[int]]:

        bitio = BitIOConst(payload)

        dct = [0.0] * 64
        dqt = fill_dqt_by_q(qf)
        prev_dc = 0.0

        strip = [[0 for _ in range(14 * 8)] for _ in range(8)]

        for m in range(14):
            zdct = [0.0] * 64
            block_ok = True

            dc_cat = self._dc_lookup[bitio.peek_bits(16)]
            if dc_cat == -1:
                break

            bitio.advance_bits(DC_CAT_OFF[dc_cat])
            n = bitio.fetch_bits(dc_cat)

            zdct[0] = map_range(dc_cat, n) + prev_dc
            prev_dc = zdct[0]

            k = 1
            while k < 64:
                ac = self._ac_lookup[bitio.peek_bits(16)]
                if ac == -1:
                    block_ok = False
                    break

                ac_len = self._ac_table[ac].length
                ac_run = self._ac_table[ac].run
                ac_size = self._ac_table[ac].size
                bitio.advance_bits(ac_len)

                # EOB
                if ac_run == 0 and ac_size == 0:
                    while k < 64:
                        zdct[k] = 0.0
                        k += 1
                    break

                # ZRL = 16 zeros
                if ac_run == 15 and ac_size == 0:
                    if k + 16 > 64:
                        block_ok = False
                        break
                    for _ in range(16):
                        zdct[k] = 0.0
                        k += 1
                    continue

                # normal run
                if k + ac_run > 64:
                    block_ok = False
                    break

                for _ in range(ac_run):
                    zdct[k] = 0.0
                    k += 1

                if ac_size != 0:
                    n = bitio.fetch_bits(ac_size)
                    if k >= 64:
                        block_ok = False
                        break
                    zdct[k] = float(map_range(ac_size, n))
                    k += 1

            if not block_ok:
                break

            for i in range(64):
                dct[i] = float(zdct[ZIGZAG[i]] * dqt[i])

            img_dct = self.filt_idct8x8(dct)

            x_offset = m * 8
            for i in range(64):
                t = int(round(img_dct[i] + 128.0))
                if t < 0:
                    t = 0
                if t > 255:
                    t = 255

                row = i // 8
                col = i % 8
                strip[row][x_offset + col] = t

        return strip

# Convenience function (stateless entrypoint)
_decoder_singleton: Image | None = None


def decode_14_blocks(payload: bytes, qf: float) -> Tuple[List[List[int]], float]:
    global _decoder_singleton
    if _decoder_singleton is None:
        _decoder_singleton = Image()
    return _decoder_singleton.decode_14_blocks(payload=payload, qf=qf)

