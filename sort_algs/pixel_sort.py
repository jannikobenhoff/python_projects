import pygame
import time
import sys, os
import random
import time

width = 800
index = 160
window = pygame.display.set_mode((width, width+100))
pygame.display.set_caption("SORT")

RED = (255, 1, 0)
GREEN = (0, 255, 0)
BLUE = (0, 2, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (125, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

color_list = [TURQUOISE, ORANGE, PURPLE, BLACK, WHITE, YELLOW, BLUE, GREEN, RED]
block_list = []
button_list = []

class Block(object):
    def __init__(self, x, y, color, height):
        self.height = height
        self.color = color
        self.y = y
        self.x = x
        self.color_int = self.color[0] + self.color[1] + self.color[2]

    def draw_block(self):
        pygame.draw.rect(window, self.color, pygame.Rect(self.x, self.y, self.height, self.height))

    def update(self):
        self.color_int = self.color[0] + self.color[1] + self.color[2]


class RefreshButton(object):
    def __init__(self, height):
        self.height = height
        self.width = height
        self.x = 20
        self.y = 820
        self.rect = pygame.Rect(self.x, self.y, self.height, self.width)

    def buttonClicked(self):
        create_block()

    def draw(self):
        pygame.draw.rect(window, BLUE, (self.x, self.y, self.width, self.height))


class SortBubbleButton(object):
    def __init__(self, height):
        self.height = height
        self.width = height
        self.x = 100
        self.y = 820
        self.rect = pygame.Rect(self.x, self.y, self.height, self.width)

    def buttonClicked(self):
        sort_color1()

    def draw(self):
        pygame.draw.rect(window, ORANGE, (self.x, self.y, self.width, self.height))


class SortInvertedButton(object):
    def __init__(self, height):
        self.height = height
        self.width = height
        self.x = 180
        self.y = 820
        self.rect = pygame.Rect(self.x, self.y, self.height, self.width)

    def buttonClicked(self):
        sort_color()

    def draw(self):
        pygame.draw.rect(window, PURPLE, (self.x, self.y, self.width, self.height))


def draw_grid():
    window.fill(WHITE)
    draw_blocks()
    draw_buttons()
    for x in range(int(width/index)):
        pygame.draw.line(window, GREY, (0, x * index), (width, index * x))
        for i in range(81):
            pygame.draw.line(window, GREY, (i * index, 0), (index * i, width))
    pygame.display.update()


def draw_blocks():
    for block in block_list:
        block.draw_block()


def draw_buttons():
    for button in button_list:
        button.draw()


def sort_color1():
    start = time.time()
    sorted = False
    length = len(block_list) - 1
    while not sorted:
        sorted = True
        for x in range(length):
            if block_list[x].color_int > block_list[x + 1].color_int:
                block_list[x].color, block_list[x + 1].color = block_list[x + 1].color, block_list[x].color
                for block in block_list:
                    block.update()
                sorted = False
            draw_grid()
        length += -1
    total_time = str((time.time() - start))
    print(total_time)


def sort_color():
    start = time.time()
    length = len(block_list) - 1
    for x in range(length):
        for i in range(x, length+1):
            if block_list[x].color_int > block_list[i].color_int:
                block_list[x].color, block_list[i].color = block_list[i].color, block_list[x].color
                for block in block_list:
                    block.update()

            draw_grid()
    total_time = str((time.time() - start))
    print(total_time)

def create_block():
    block_list.clear()
    for x in range(int(width / index)):
        for y in range(int(width / index)):
            i = random.randint(0, len(color_list) - 1)
            block = Block(x * index, y * index, color_list[i], index)
            block.draw_block()
            block_list.append(block)

def main():

    create_block()

    refresh = RefreshButton(60)
    sort = SortBubbleButton(60)
    invert = SortInvertedButton(60)
    button_list.append(invert)
    button_list.append(refresh)
    button_list.append(sort)

    block = False
    run = True
    while run:
        draw_grid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    print('enter')
            if pygame.mouse.get_pressed()[0]: # LEFT
                pos = pygame.mouse.get_pos()
                for button in button_list:
                    if button.rect.collidepoint(pos):
                        button.buttonClicked()


    pygame.quit()



main()