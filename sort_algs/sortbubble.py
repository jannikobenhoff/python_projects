import random

listNumbers = []


def getNumbersInList():
    for x in range(115):
        i = random.randint(1,500)
        if i not in listNumbers:
            listNumbers.append(i)


def sort_bubble(list):
    length = len(list) -1
    sorted = False
    while not sorted:
        sorted = True
        for x in range(0,length):
            if list[x]>list[x+1]:
                list[x], list[x+1] = list[x+1], list[x]
                sorted = False


def sort_diff(list):
    length = len(list) - 1
    sorted = False
    while not sorted:
        sorted = True
        for x in range(0, length):
            for i in range(x, length):
                if list[x]>list[i]:
                    list[x], list[i] = list[i], list[x]
                    sorted = False

#sort_bubble(listNumbers)
#sort_diff(listNumbers)
#print(listNumbers)


