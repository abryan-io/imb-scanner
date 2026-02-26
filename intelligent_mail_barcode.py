#!/usr/bin/env python3
# Python 3 port of samrushing/pyimb
# Original: https://github.com/samrushing/pyimb
# License: Simplified BSD
#
# Changes from original:
#   - Python 3 compatibility (print statements → functions, has_key → in, etc.)
#   - decode() returns a dict instead of printing
#   - Leading octal literals fixed
#   - make_inverted_tabs() uses .items() not iteritems

import sys

def crc11(input):
    gen_poly = 0x0f35
    FCS = 0x07ff
    data = input[0] << 5
    for bit in range(2, 8):
        if (FCS ^ data) & 0x400:
            FCS = (FCS << 1) ^ gen_poly
        else:
            FCS = FCS << 1
        FCS &= 0x7ff
        data <<= 1
    for byte_index in range(1, 13):
        data = input[byte_index] << 3
        for bit in range(8):
            if (FCS ^ data) & 0x400:
                FCS = (FCS << 1) ^ gen_poly
            else:
                FCS = FCS << 1
            FCS &= 0x7ff
            data <<= 1
    return FCS

def reverse_int16(input):
    reverse = 0
    for i in range(16):
        reverse <<= 1
        reverse |= input & 1
        input >>= 1
    return reverse

def init_n_of_13(n, table_length):
    table = {}
    index_low = 0
    index_hi = table_length - 1
    for i in range(8192):
        bit_count = bin(i).count('1')
        if bit_count != n:
            continue
        reverse = reverse_int16(i) >> 3
        if reverse < i:
            continue
        if i == reverse:
            table[index_hi] = i
            index_hi -= 1
        else:
            table[index_low] = i
            index_low += 1
            table[index_low] = reverse
            index_low += 1
    if index_low != index_hi + 1:
        raise ValueError(index_low, index_hi)
    return table

def make_inverted_tabs():
    global inverted
    inverted = {}
    for k, v in tab5.items():
        if v in inverted:
            raise ValueError
        inverted[v] = (0, k)
    for k, v in tab2.items():
        if v in inverted:
            raise ValueError
        inverted[v] = (1, k)

def binary_to_codewords(n):
    r = []
    n, x = divmod(n, 636)
    r.append(x)
    for i in range(9):
        n, x = divmod(n, 1365)
        r.append(x)
    r.reverse()
    return r

def codewords_to_binary(codes):
    n = 0
    cr = codes[:]
    for code in cr[:-1]:
        n = (n * 1365) + code
    n = (n * 636) + cr[-1]
    return n

def convert_routing_code(zip):
    if len(zip) == 0:
        return 0
    elif len(zip) == 5:
        return int(zip) + 1
    elif len(zip) == 9:
        return int(zip) + 100000 + 1
    elif len(zip) == 11:
        return int(zip) + 1000000000 + 100000 + 1
    else:
        raise ValueError(zip)

def unconvert_routing_code(n):
    if n > 1000000000:
        return n - (1000000000 + 100000 + 1)
    elif n > 100000:
        return n - (100000 + 1)
    elif n:
        return n - 1
    else:
        return 0

def convert_tracking_code(enc, track):
    assert(len(track) == 20)
    enc = (enc * 10) + int(track[0])
    enc = (enc * 5) + int(track[1])
    for i in range(2, 20):
        enc = (enc * 10) + int(track[i])
    return enc

def unconvert_tracking_code(n):
    r = []
    for i in range(2, 20):
        n, x = divmod(n, 10)
        r.append(x)
    n, x = divmod(n, 5)
    r.append(x)
    n, x = divmod(n, 10)
    r.append(x)
    r.reverse()
    return n, ''.join([str(int(x)) for x in r])

def to_bytes(val, nbytes):
    r = []
    for i in range(nbytes):
        r.append(val & 0xff)
        val >>= 8
    r.reverse()
    return r

def encode(barcode_id, service_type_id, mailer_id, serial, delivery):
    n = convert_routing_code(delivery)
    if str(mailer_id)[0] == '9':
        tracking = '%02d%03d%09d%06d' % (barcode_id, service_type_id, mailer_id, serial)
    else:
        tracking = '%02d%03d%06d%09d' % (barcode_id, service_type_id, mailer_id, serial)
    n = convert_tracking_code(n, tracking)
    fcs = crc11(to_bytes(n, 13))
    codewords = binary_to_codewords(n)
    codewords[9] *= 2
    if fcs & (1 << 10):
        codewords[0] += 659
    r = []
    for b in codewords:
        if b < 1287:
            r.append(tab5[b])
        elif 127 <= b <= 1364:
            r.append(tab2[b - 1287])
        else:
            raise ValueError
    for i in range(10):
        if fcs & 1 << i:
            r[i] = r[i] ^ 0x1fff
    return make_bars(r)

def make_bars(code):
    r = []
    for i in range(65):
        index, bit = tableA[i]
        ascend  = (code[index] & (1 << bit) != 0)
        index, bit = tableD[i]
        descend = (code[index] & (1 << bit) != 0)
        r.append('TADF'[descend << 1 | ascend])
    return ''.join(r)

def unbar(code):
    assert(len(code) == 65)
    r = [0] * 10
    for i in range(65):
        ch = code[i]
        ia, ba = tableA[i]
        id_, bd = tableD[i]
        if ch == 'A':
            r[ia] |= 1 << ba
        elif ch == 'D':
            r[id_] |= 1 << bd
        elif ch == 'F':
            r[ia] |= 1 << ba
            r[id_] |= 1 << bd
        else:
            pass
    return r

def decode(codes):
    """
    Decode a 65-character FADT string into IMB components.
    Returns a dict with keys: tracking, routing, barcode_id, service_type,
    mailer_id, serial, raw_number. Returns None on failure.
    """
    if len(codes) != 65:
        return None
    fcs = 0
    codes_arr = unbar(codes)
    r = []
    for i in range(10):
        code = codes_arr[i]
        if code not in inverted:
            code = code ^ 0x1fff
            fcs |= 1 << i
        if code not in inverted:
            return None  # unrecoverable decode error
        bump, val = inverted[code]
        if bump:
            val += 1287
        r.append(val)
    if r[0] > 659:
        fcs |= 1 << 10
        r[0] -= 659
    r[9] >>= 1
    binary = codewords_to_binary(r)
    fcs0 = crc11(to_bytes(binary, 13))

    a, tracking = unconvert_tracking_code(binary)
    routing_num = unconvert_routing_code(a)
    routing = '%d' % routing_num

    barcode_id   = tracking[0:2]
    service_type = tracking[2:5]

    if tracking[5] == '9':
        mailer_id = tracking[5:14]
        serial    = tracking[14:20]
    else:
        mailer_id = tracking[5:11]
        serial    = tracking[11:20]

    # Build routing string with correct zero-padding
    rlen = len(routing)
    if routing_num == 0:
        routing_str = ''
    elif routing_num <= 99999:          # ZIP only (5 digits)
        routing_str = routing.zfill(5)
    elif routing_num <= 999999999:      # ZIP+4 (9 digits)
        routing_str = routing.zfill(9)
    else:                               # ZIP+4+DPC (11 digits)
        routing_str = routing.zfill(11)

    return {
        'raw_number':   str(binary),
        'tracking':     tracking,
        'routing':      routing_str,
        'barcode_id':   barcode_id,
        'service_type': service_type,
        'mailer_id':    mailer_id,
        'serial':       serial,
        'crc_ok':       (fcs == fcs0),
    }

def process_bar_table():
    global tableA, tableD
    tableA = {}
    tableD = {}
    for i in range(65):
        entry = bar_table[i]
        i0, d, i1, a = entry.split()
        i0 = ord(i0) - 65
        i1 = ord(i1) - 65
        d  = int(d)
        a  = int(a)
        tableD[i] = i0, d
        tableA[i] = i1, a

bar_table = [
    'H 2 E 3', 'B 10 A 0', 'J 12 C 8', 'F 5 G 11', 'I 9 D 1',
    'A 1 F 12', 'C 5 B 8', 'E 4 J 11', 'G 3 I 10', 'D 9 H 6',
    'F 11 B 4', 'I 5 C 12', 'J 10 A 2', 'H 1 G 7', 'D 6 E 9',
    'A 3 I 6', 'G 4 C 7', 'B 1 J 9', 'H 10 F 2', 'E 0 D 8',
    'G 2 A 4', 'I 11 B 0', 'J 8 D 12', 'C 6 H 7', 'F 1 E 10',
    'B 12 G 9', 'H 3 I 0', 'F 8 J 7', 'E 6 C 10', 'D 4 A 5',
    'I 4 F 7', 'H 11 B 9', 'G 0 J 6', 'A 6 E 8', 'C 1 D 2',
    'F 9 I 12', 'E 11 G 1', 'J 5 H 4', 'D 3 B 2', 'A 7 C 0',
    'B 3 E 1', 'G 10 D 5', 'I 7 J 4', 'C 11 F 6', 'A 8 H 12',
    'E 2 I 1', 'F 10 D 0', 'J 3 A 9', 'G 5 C 4', 'H 8 B 7',
    'F 0 E 5', 'C 3 A 10', 'G 12 J 2', 'D 11 B 6', 'I 8 H 9',
    'F 4 A 11', 'B 5 C 2', 'J 1 E 12', 'I 3 G 6', 'H 0 D 7',
    'E 7 H 5', 'A 12 B 11', 'C 9 J 0', 'G 8 F 3', 'D 10 I 2',
]

# Module-level initialization
process_bar_table()
tab5 = init_n_of_13(5, 1287)
tab2 = init_n_of_13(2, 78)
make_inverted_tabs()
