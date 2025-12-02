import pymcprotocol as mc

# 문자열 -> 워드 리스트 함수
def str_to_word(string):
    temp = string.encode("ascii")
    if len(temp) % 2 != 0:
        temp += b"\x00"
    words = []
    for i in range(0, len(temp), 2):
        word = temp[i] + (temp[i+1] << 8)
        words.append(word)
    return words

# 문자열 복원 함수
def word_to_str(words):
    byte_array = bytearray()
    for word in words:
        byte_array.append(word & 0xFF)
        byte_array.append((word >> 8) & 0xFF)
    return byte_array.decode("ascii").rstrip("\x00")

# 문자열 -> 워드 리스트
words = str_to_word("stop")

# PLC 연결
plc = mc.Type3E()
plc.setaccessopt(commtype="binary")
plc.connect("192.168.3.120", 1025)
print("연결 성공")

# 초기화
plc.batchwrite_wordunits("D100", [0, 0, 0])
plc.batchwrite_bitunits("M100", [0])

# 워드 쓰기
plc.batchwrite_wordunits("D100", words)

# 워드 읽기
vals = plc.batchread_wordunits("D100", len(words))
print("D100 words:", vals)

# 비트 쓰기
plc.batchwrite_bitunits("M100", [1])

# 비트 읽기
bit_vals = plc.batchread_bitunits("X7", 1)
print("M100 bit:", bit_vals)

# 문자열 복원
recovered = word_to_str(vals)
print("복원된 문자열:", recovered)

plc.close()