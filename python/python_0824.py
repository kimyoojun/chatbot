# def print_3_times():
#     print("안녕하세요")
#     print("안녕하세요")
#     print("안녕하세요")

# print_3_times()



# def print_n_times(value, n):
#     for i in range(n):
#         print(value)

# print_n_times("ㅎㅇㅎㅇ", 3)



# def print_n_times(n, *values):
#     for i in range(n):
#         for value in values:
#             print(value)
#         print()

# print_n_times(3, "안녕하세요", "즐거움", "졸려") 



# def print_n_times(value, n=2):
#     for i in range(n):
#         print(value)

# print_n_times("안녕하세요", )



# def print_n_times(*values, n=2):
#     for i in range(n):
#         for value in values:
#             print(value)
#         print()

# print_n_times("안녕하세요", "즐거운", n=3)



# def return_test():
#     return 100000000000000000000000

# value = return_test()
# print(value)



# def sum_all(start, end, step=1):
#     output = 0

#     for i in range(start, end + 1, step):
#         output += i
#     return output

# output = sum_all(0, 100, 10)
# print(f" 0 to 100{output}")
# print(f" 0 to 1000 {sum_all(end = 100)}")
# print(f" 0 to 1000 {sum_all(end = 100, step = 2)}")



# def mul(*values):

#     output = 1
#     for v in values:
#         output *= v

#     return output
        
# print(mul(5, 7, 9, 10))



# file = open('basic.txt', "w")

# file.write("Hello Python Programming...!")

# file.close()



# with open("basic.txt", "r") as file:

#     contents = file.read()
# print(contents)



# import random

# hanguls = list("가나다라마바사아자차카타파하")

# with open("info.txt","w") as file:
#     for i in range(1000):
#         name = random.choice(hanguls) + random.choice(hanguls)
#         weight = random.randrange(40, 100)
#         height = random.randrange(140, 200)

#         file.write(f"{name}, {weight}, {height}")
    


# with open("info.txt", "r") as file:
#     for line in file:
#         (name, weight, height) = line.strip().split(", ")

#         if (not name) or (not weight) or (not height):
#             continue
        
#         bmi = int(weight) / ((int(height) / 100) ** 2)
#         result = ""
#         if 25 <= bmi:
#             result = "과체중"
#         elif 18.5 <= bmi:
#             result = "정상 체중"
#         else:
#             result = "저체중"

#         print('\n'.join([
#             "이름: {}",
#             "몸무게: {}",
#             "키: {}",
#             "BMI: {}",
#             "결과: {}"
#         ]).format(name, weight, height, bmi, result))
#         print()



# treehit = 0

# while treehit < 10:
#     treehit = treehit + 1
#     print("나무를 {}번 찍었습니다".format(treehit))
#     if treehit == 10:
#         print("나무 넘어갑니다")



# prompt = """
# 1.Add
# 2.Del
# 3.List
# 4.Quit

# ...Enter number:
# """

# number = 0
# while number != 4:
#     print(prompt)
#     number = int(input())



# a = 0
# while a < 10:
#     a += 1
#     print(a)


# b = 10
# while b > 0:
#     print(b)
#     b -= 1



# for i in range(1,11):
#     print(i)

# for i in range(10,0,-1):
#     print(i)



# a = """
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 10
# 9
# 8
# 7
# 6
# 5
# 4
# 3
# 2
# 1
# """
# print(a)



# coffee = 10
# money = 300
# while money:
#     print("돈을 받았으니 커피를 줍니다.")
#     coffee = coffee -1
#     print("남은 커피의 양은 %d개 입니다." % coffee)
#     if not coffee:
#         print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
#         break



# coffee = 10
# while True:
#     money = int(input("돈을 넣어 주세요: "))
#     if money == 300:
#         print("커피를 줍니다.")
#         coffee = coffee -1
#     elif money >= 300:
#         print("커피를주고 거스름돈{}원을 줍니다.".format(int(money) - int(300)))
#         coffee = coffee -1
#     else:
#         print('돈을 다시 돌려주고 커피를 주지 않습니다.')
#     if not coffee:
#         print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
#         break
#     print("남은 커피의양은 %d개입니다." % coffee)
        




# for i in range(1,101):
#     if i % 2 == 0:
#         print(i)



# for i in range(2,101,2):
#     print(i)



# import random

# a = 0
# b = 0
# c = 0
# e = 0
# for g in range(1,101):
#     d = random.randrange(1,5)
#     if d == 1 :
#         print("꼬북이")
#         a += 1
#     elif d == 2:
#         print("파이리")
#         b += 1
#     elif d == 3:
#         print("피카츄")
#         c += 1
#     else :
#         print("메타몽")
#         e += 1
# print("꼬북이는 {}마리 나왔습니다".format(a))
# print("파이리는 {}마리 나왔습니다".format(b))
# print("피카츄는 {}마리 나왔습니다".format(c))
# print("메타몽은 {}마리 나왔습니다.".format(e))

# a2 = 0
# b2 = 0
# c2 = 0
# for i in range(e):
#     f = random. randrange(1,4)
    
#     if f == 1 :
#         print("꼬북이")
#         a2 += 1
#     elif f == 2:
#         print("파이리")
#         b2 += 1
#     else:
#         print("피카츄")
#         c2 += 1

# print("메타몽 {}마리가 꼬북이로 변했습니다".format(a2))
# print("메타몽 {}마리가 파이리로 변했습니다".format(b2))
# print("메타몽 {}마리가 피카츄로 변했습니다".format(c2))
# print()
# print("꼬북이는 총 {}마리가 나왔습니다".format(int(a) + int(a2)))
# print("파이리는 총 {}마리가 나왔습니다".format(int(b) + int(b2)))
# print("피카츄는 총 {}마리가 나왔습니다".format(int(c) + int(c2)))



# import random
# a = input("수컷의 애칭을 지어주세요> ")
# b = input("암컷의 애칭을 지어주세요> ")
# c = 1
# day = 1

# while c <= 500:
#     for f in range(c):
#         d = random.randrange(1,6)
#         print("새끼가 {}마리 태어났습니다.".format(int(d) * int(2)))
#         c = c + d
#     if day >= 2:
#         e = random.randrange(1,6)
#         print("암수 {}쌍이 죽었습니다".format(e))
#         c -= e
   
#     day += 1
# print("{}턴후에 어항이 가득 찼습니다".format(day))
# print("어항에는 {}쌍에 금붕어가 있습니다".format(c))



# a = 1

# for i in range(10, 0, -1):
#     print(i * " ", end = ' ')
#     print(a * '*')
#     a +=2

# a = 21

# for i in range(11):
#     print(i * " ", end = ' ')
#     print(a * '*')
#     a -=2




# a = 1
# b = 9
# c = 10

# for i in range(10):
#     print(a * "* ", end = '')
#     print(b * "  ", end = '')
#     print(c * "* ")
#     a += 1
#     b -= 1
#     c -= 1



# a = 9
# b = 1
# c = 0
# d = 10

# for i in range(10):
#     print(a * "  ", end = '')
#     print(b * "* ", end = '')
#     print(c * "  ", end = '')
#     print(d * "* ")
#     a -= 1
#     b += 1
#     c += 1
#     d -= 1



# import random

# gu1 = 0
# gu = 0
# b = 1
# c=1

# print("{}번 반복".format(o))
# while b <= 10:
#     youtube = "ㅁ"
#     con = "ㅠ"
#     print("방송 송출 ~ !")
#     second = 1
#     si = 0
#     gu = 0
#     b += 1

#     while second < 10:
#         a = random.randrange(2500, 2501)
#         si += a
#         second += 1

#     print("10초동안 시청자수 {}명".format(si))
#     a = random.randrange(1, 6)
#     gu = si / a
#     gu1 = gu1 + gu
#     print("증가한 구독자 수 {}명".format(gu))
#     print("총 구독자수{}명".format(gu1))

# if gu1 >= 100000:
#     print("실버버튼 증정 성공")
# else:
#     print("유튜브 채널 폭파")
#     print("ㅎㅎ")

