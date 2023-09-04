# list_a = [0, 1, 2, 3, 4, 5]
# print("# 리스트의 요소 하나 제거하기")

# del list_a[1]
# print("del list_a[1]:", list_a)

# list_a.pop(2)
# print("pop(2):", list_a)



# array = [273, 32, 103, 57, 52]

# for element in array:

#     print(element)



# numbers = [273, 103, 5, 32, 65, 9, 72, 800, 99]

# for number in numbers:
#     if number > 100:
#         print("-100 이상의 수:", number)



# numbers = [273, 103, 5, 32, 65, 9, 72, 800, 99]

# for number in numbers:
#     if number % 2 == 0:
#         print(number, "는 짝수입니다.")
#     else:
#         print(number, "는 홀수입니다.")



# numbers = [273, 103, 5, 32, 65, 9, 72, 800, 99]
# for number in numbers:
#     if number >= 100:
#         print(number, "는 3 자릿수입니다.")
#     elif number >= 10:
#         print(number, "는 2자릿수입니다.")
#     else:
#         print(number, "는 1자릿수입니다.")




# numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# output = [[], [], []]

# for number in numbers:
#     # print(number % 3)
#     output[number % 3 -1 ].append(number)

# print(output)
    



# numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# for i in range(0, len(numbers) // 2):

#     j = 
#     print(f"i = {i}, j = {j}")
#     numbers[j] = numbers[j] ** 2



# for i in range(2,10):
#     for j in range(1,10):
#         print(f"{i} * {j} = { i * j }", end=' ')
#     print('')

# print()


# a = [1,2,3,4]
# result = []

# for num in a:
#     result.append(num * 3)

# print(result)

# result = []
# result = [num * 3 for num in a]

# print(result)





# print("1부터 10까지 출력하기(1)")

# i = 1
# while i <= 10:
#      print(i)
#      i += 1





# print("1부터 10까지 출력하기(2)")

# for i in range(1,11):
#     print(i)
    

# i = 10
# print("10부터 1까지 출력하기")
# while i >= 1:
#     print(i)
#     i -= 1



# print("1부터 100까지의 합 구하기")
# sum = 0
# i = 1
# while i <= 100:
#     sum +=i
#     i+=1

# print(sum)



# print("1부터 100까지의 수 중 짝수의 합 구하기")
# sum = 0
# i = 2
# while i <= 100:
#     sum += i
#     i +=2

# print(sum)
    


# print("1부터 100까지의 수 중 짝수의 합 구하기(2)")
# sum = 0
# i = 1
# while i <= 100:
#     if i % 2 == 0:
#         sum += i
#     i+=1

# print(sum)



# print("예제 23")

# sum = 0
# a=1
# while a <=100:
#     if a % 2 == 0:
#         sum-=a
#     else:
#         sum+=a
#     a+=1
# print(sum)



# print("예제 24")
# tact = 1
# i = 5
# for i in range(5,1,-1):
#     tact = tact * i

# print(tact)


# print("예제 25")

# n = int(input("수를 입력하세요"))
# i = 1
# while n>=i:
#     if n%i == 0:
#         print(i)
#     i+=1


# n = int(input("수를 입력해주세요> "))
# for i in range(1,n,1):
#     if n%i==0:
#         print(i)
    
        

# sum = 0
# for i in range(1,101):
#     if i % 2 == 0:
#         sum = sum - i
#     else:
#         sum = sum + i
# print(sum)






# # 구구단 예제


# for i in range(1,10):
#     print()
#     for j in range(2,10):
#         print(f"{j}*{i}={j*i}\t", end='')
    


    
# dict_a = {
#     "name":"어벤저스 엔드게임",
#     "type":"히어로 무비"
# }

# print(dict_a)
# print(dict_a["name"])
# print(dict_a["type"])

# dict_b = {
#     "director":["안소니 루소","조 루소"],
#     "cast":["아이언맨","타노스","토르","닥터스트레인지","헐크"]
# }

# print(dict_b["director"])
# print(dict_b["cast"])