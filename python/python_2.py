# j = 0
# i = 0
# while i < 10:
#     j = 0
#     print(f"i = {i}")
#     while j < 10:
#         print(f"j = {j}\t", end='')
#         j += 1
#     print()
#     i += 1


i = 2
j = 1
while i < 10:
    j = 1
    while j < 10:
        print("{} * {} = {}\t".format(i, j , int(i)*int(j)), end='')
        j += 1
    print()
    i += 1
