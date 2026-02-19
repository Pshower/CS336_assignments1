# readme: I will use || to include my answer.

test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")

print(utf8_encoded)
print(type(utf8_encoded))
list(utf8_encoded)

print(len(test_string))
print(len(utf8_encoded))
print(utf8_encoded.decode("utf-8"))

# a
# | UTF-8 can cover most charaters. UTF-16 or UTF-32 takes more bytes. |
utf16_encoded = test_string.encode("utf-16")

print(utf16_encoded)
print(type(utf16_encoded))
list(utf16_encoded)

print(len(test_string))
print(len(utf16_encoded))
print(utf16_encoded.decode("utf-16"))

# b
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
# print(decode_utf8_bytes_to_str_wrong(utf8_encoded))
print([b for b in utf8_encoded])
print(utf8_encoded.__repr__())
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data
error_encode = "に".encode("utf-8")
print([b for b in error_encode])
# | Because utf-8 use 1-4 nums' vector to encode Japanes. List formation split vector. |
# print("0xe3".decode("utf-8"))

# c
for i in range(256):
    for j in range(256):
        print(f"[{i}, {j}]: {bytes([i, j]).decode('utf-8')}")
[0, 128]