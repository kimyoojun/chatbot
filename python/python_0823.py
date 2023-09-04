# dictionary = {
#     "name": "7D 건조 망고",
#     "type": "당절임",
#     "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
#     "origin": "필리핀"
# }

# print("name:", dictionary["name"])
# print("type:", dictionary["type"])
# print("ingredient:", dictionary["ingredient"])
# print("origin:", dictionary["origin"])
# print()

# dictionary["name"] = "8D 건조 망고"
# print("name:", dictionary["name"])  

# import random
##임의의 실수를 반환한다.
# print(random.random())

##a~b사이의 숫자를 반환한다.
# print(random.randint(0, 2))
# print(random.randrange(0, 3))

#로또 번호 뽑기 6개, 청소 당번 번호 뽑기
# import random
# numbers = []
# while len(numbers) < 7:
#     number = random.randint(1, 10)
#     if number not in numbers:
#         numbers.append(number)

# print(numbers)
    


# dictionary = {
#     "name": "7D 건조 망고",
#     "type": "당절임",
#     "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
#     "origin": "필리핀"
# }

# key = input("> 접근하고자 하는 키:")

# if key in dictionary:
#     print(dictionary[key])
# else:
#     print("존재하지 않는 키에 접근하고 있습니다")



# dictionary = {
#     "name": "7D 건조 망고",
#     "type": "당절임",
#     "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
#     "origin": "필리핀"
# }

# value = dictionary.get("존재하지 않는 키")
# print("값:", value)

# if value == None:
#     print("존재하지 않는 키에 접근했었습니다.")



# dictionary = {
#     "name": "7D 건조 망고",
#     "type": "당절임",
#     "ingredient": ["망고", "설탕", "메타중아황산나트륨", "치자황색소"],
#     "origin": "필리핀"
# }

# for key in dictionary:
#     print(key, ":", dictionary[key])

# pets = [
#     {"name": "구름", "age": 5},
#     {"name": "초코", "age": 3},
#     {"name": "이지", "age": 1},
#     {"name": "호랑이", "age": 1},
# ]

# print("# 우리 동네 애완 동물들")
# for key in pets:
#     print(key["name"], end = ' ')
#     print(key["age"], end = '')
#     print("살")



# numbers = [1,2,6,84,3,2,1,9,5,4,9,7,2,1,3,5,4,8,9,7,2,3]
# counter = {}

# for number in numbers:
#     if number in counter:
#         counter[number] += 1
#     else:
#         counter[number] = 1

# print(counter)



# character = {
#     "name": "기사",
#     "lever": 12,
#     "items": {
#         "swore": "불꽃의 검",
#         "armor": "풀플레이트"
#     },
#     "skill": ["베기", "세게 베기", "아주 세게 베기"]
# }

# for key in character:
#     if type(character[key]) is dict:
#         dicdic = character[key]
#         for k in dicdic:
#             print(k, ":", dicdic[k])
#     elif type(character[key]) is list:
#         diclis = character[key]
#         for k in diclis:
#             print(key,":",k)
#     else:
#         print(key,":", character[key])



# array = [273, 32, 103, 57, 52]

# for i in range(len(array)):
#     print(f"{i}번째 반복문 :  {array[i]}")
    


# for i in range(4, - 1, -1):
#     print("현재 반복 변수: {}".format(i))



# numbers = [5,15,6,20,7,25]

# for number in numbers:
#     if number < 10:
#         continue

#     print(number)



# key_list = ["name", "hp", "mp", "level"]
# value_list = ["기사", 200, 30, 5]
# character = {}

# for i in range(len(key_list)):
#     k = key_list[i]
#     v = value_list[i]
#     character[k] = v

# print(character)



# limit = 10000
# i = 1
# sum_value = 0

# while sum_value < 10000:
#     sum_value += i
#     i+=1
    
# print("{}를 더할 때 {} 을 넘으며 그때의 값은 {}입니다.".format(i-1,limit, sum_value))




# limit = 10000
# i = 1
# sum_value = 0
# j = 2
# while sum_value < limit:
#     sum_value = 0
#     i = 0
#     for i in range(i,j + 1):
#         i += 1
#         sum_value += 1
#     j += 1
# print(j,limit,sum_value)



# max_value = 0
# a = 0
# b = 0

# for i in range(1,100+1):
#     j = 100 - i

#     if i * j > max_value :
#         max_value = i * j
#         a = i
#         b = j 

# print("최대가 되는 경우: {} * {} = {}".format(a, b, max_value))



# example_list = ["요소A", "요소B", "요소C"]

# print("# 단순 출력")
# print(example_list)
# print()

# print("# enumerate() 함수 적용 출력")
# print(enumerate(example_list))
# print()

# print("# list() 함수로 강제 변환 출력")
# print(list(enumerate(example_list)))
# print()

# for i, value in enumerate(example_list):
#     print(f"{i}번째 요소는 {value}입니다.")




# example_dictionary = {
#     "키A":"값A",
#     "키B":"값B",
#     "키C":"값C"
# }

# print("딕셔너리의 items() 함수")

# print(example_dictionary.items())
# print()

# for key, element in example_dictionary.items():
#     print(f"dictionaru[{key}] = {element}")


# array = []
# for i in range(0,20,2):
#     array.append(i*i)

# print(array)


# array = [i * i for i in range(0,20,2)]
# print(array)



# array = ["사과","자두","초콜릿","바나나","체리"]
# ouput = []

# for fruit in array:
#     if fruit != "초콜릿":
#         ouput.append(fruit)

# print(ouput)

# ouput = [fruit for fruit in array if fruit != "초콜릿"]

# print(ouput)



# numbers = [1, 2, 3, 4, 5, 6]
# r_num = reversed(numbers)

# print("reversed_numbers :", r_num)
# print(next(r_num))
# print(next(r_num))
# print(next(r_num))
# print(next(r_num))
# print(next(r_num))


# food = 'python\s favorite food is per1'
# say = "\"Python is very easy/\" he says."

# print(food)
# print(say)



# print("=" * 50)
# print("my program")
# print("=" * 50)



# a = "Life is Too Short, You Need Python"
# print(a[3])
# print(a[0:4])
# print(a[5:-7])



# num = 3
# print("I eat %d apples"%3)
# print("I eat {} apples".format(num))
# print(f"I eat {num} apples")



# a = ","
# print(a.join('abcd'))



# a = " hi "
# print(a.strip())



# a = "Life is too short"
# b = a.replace("Life", "Your lng")



# add = [1,2,3,['a','b','c']]

# print(add[:2])
# print(add[2:])
# print(add[3])



# a = [1,2,3]

# a[2] = 4
# print(a)
# a[1:2] = ['a','b','c']
# print(a)
# a[1:3] = []
# print(a)
# del a[1]
# print(a)


# a = [1,5,4,2,3]
# a.sort()
# print(a)



# a = int(input("첫번째 숫자를 입력하세요> "))
# b = int(input("두번째 숫자를 입력하세요>"))

# print(a + b)
# print(a * b)
# print(a / b)
# print(a % b)


# x = 3
# y = 2
# print(x > y)
# print(x > y)

# if x > y:
#     print("참")
# else:
#     print("거짓")




# a = int(input("첫번째 숫자를 입력하세요> "))
# if a > 0:

#     b = int(input("두번째 숫자를 입력하세요>"))
#     if b > 0:
#         print(a + b)
#         print(a * b)
#         print(a / b)
#         print(a % b)
        
#     else:
#         print("error")

# else:
#     print("error")



# money = 500000000000000000000000
# card = 1

# if money >= 3000 or card:
#     print("택시타고 가")
# else:
#     print("걸어가")



# pocket = ['paper', 'cellphone', 'money']
# del pocket[2]
# print(pocket)

# if 'money' in pocket:
#     print("택시를 타고 가라")
# else:
#     print("걸어가라")