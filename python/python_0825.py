# marks = [90,25,67,45,80]

# number = 0

# for mark in marks:
#     number = number + 1
#     if mark < 60:
#         continue
#     print(f"{number}학생은 합격입니다.")



# for i in range(2,10):
#     for j in range(1,10):
#         print(f"{i} * {j} = {i * j} ",end='')
#     print('')



# a = [1,2,3,4]
# result = []

# for num in a:
#     if num % 2 == 0:
#         result.append(num * 3)
# print(result)
    
# a= [1,2,3,4]
# result = [num * 3 for num in a if num % 2 == 0]
# print(result)



# i = 0
# while True:
#     i += 1
#     if i > 5: break
#     print("*" * i)



# A = [70,60,55,75,95,90,80,80,85,100]
# total = 0
# for score in A:
#     total += score

# average = total / len(A)
# print(average)



# def sum_mul(choice,*args):
#     if choice =="sum":
#         result = 0
#         for i in args:
#             result = result + i
#     elif choice == "mul":
#         result = 1
#         for i in args:
#             result = result * i
#     return result

# result = sum_mul('sum',1,2,3,4,5)
# print(result)
# result = sum_mul('mul',1,2,3,4,5)
# print(result)




# a=[]

# for i in range(1,11,1):
#     a.append(i * 10)

# print(a)
# for i in range(10,0,-1):
#     print(a[i-1])

    

# money = input("돈을 넣어 주세요 > ")

# a = []

# menu = input("""
# 음료를 선택해 주세요

# 1) 콜라 1200
# 2) 우주맛 콜라 1900
# 3) 제로콜라 1200
# 4) 스프라이트 1100
# 5) 환타 900
# 6) 닥터페퍼 1100
# 7) 몬스터 1800
# 8) 파워에이드 1900
# 9) 네스티 1600
# 10) 글라소 비타민 워터 2100
# 11) 미닛메이드 1700
# 12) 조지아 커피 900
# 13) 암바사 900
# 14) 마테차 1700
# S) 계산하기
# X) 종료하기""")

# print(menu)


# a.append(menu)



# def factorial(n):
    
#     if n == 0:
#         return 1
#     else:
#         return n * factorial(n-1)
    
# print("1!", factorial(1))
# print("2!", factorial(2))
# print("3!", factorial(3))
# print("4!", factorial(4))



# counter = 0

# def fibonacci(n):
#     global counter
#     counter += 1
#     if n == 1:
#         return 1
#     if n == 2:
#         return 1
#     else: 
#         return fibonacci(n-1) + fibonacci(n-2)

# print(fibonacci(5))



# dictionary = {
#     1: 1,
#     2: 1
# }

# def fibonacci(n):
#     if n in dictionary:
#         return dictionary[n]
#     else:
#         output = fibonacci(n-1) + fibonacci(n - 2)
#         dictionary[n] = output
#         return output
    
# print(fibonacci(10))
# print(fibonacci(20))
# print(fibonacci(30))
# print(fibonacci(35))




# def flatten(data):
#     result = []
#     if type(data) is list:
#         for el in data:
#             result += flatten(el)
#     else:
#         result += [data]
#     return result

# example = [[1,2,3], [4,[5,6]],7,[8,9]]
# print(example)
# print(flatten(example))



# a, b = 10, 20
# print("# 교환 전 값")
# print("a:", a)
# print("b:", b)
# print()

# a, b = b, a
# print("# 교환 후 값")
# print("a:", a)
# print("b:", b)
# print()



# a, b = 97, 40
# print(divmod(a,b))
# x, y = divmod(a,b)
# print(x)
# print(y)



# def call_10_times(func):
#     for i in range(10):
#         func()

# def print_hello():
#     print("안녕하세요")

# call_10_times(print_hello)



# def power(item):
#     return item * item

# def under_3(item):
#     return item < 3

# list_input_a = [1, 2, 3, 4, 5]

# output_a = map(power, list_input_a)
# print("map() 함수의 실행결과")
# print(output_a)
# print(list(output_a))
# print()

# output_b = filter(under_3, list_input_a)
# print(output_b)
# print(list(output_b))



# power = lambda x: x * x
# under_3 = lambda x: x < 3

# list_input_a = [1, 2, 3, 4, 5]



