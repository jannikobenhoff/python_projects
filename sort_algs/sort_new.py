import pygame
import time
import sys, os
import random

width = 800
window = pygame.display.set_mode((width, width+100))
pygame.display.set_caption("SORT")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

block_list = []
button_list = []

class Block(object):
    def __init__(self, x):
        i = random.randint(0, 75)
        self.height = i * 10
        self.color = RED
        self.y = width - self.height
        self.x = x
        self.rect = pygame.Rect(self.x, self.y, 20, self.height)

    def draw_block(self):
        pygame.draw.rect(window, self.color, pygame.Rect(self.x, self.y, 20, self.height))


class RefreshButton(object):
    def __init__(self, height):
        self.height = height
        self.width = height
        self.x = 20
        self.y = 820
        self.rect = pygame.Rect(self.x, self.y, self.height, self.width)

    def buttonClicked(self):
        for block in block_list:
            i = random.randint(0, 75)
            block.height = i * 10
            block.y = width - block.height

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
        bubble_sort(draw_grid)

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
        invert_sort(draw_grid)

    def draw(self):
        pygame.draw.rect(window, PURPLE, (self.x, self.y, self.width, self.height))


def draw_grid():
    window.fill(WHITE)
    draw_blocks()
    draw_buttons()
    for x in range(81):
        pygame.draw.line(window, GREY, (0, x * 10), (width, 10 * x))
        for i in range(81):
            pygame.draw.line(window, GREY, (i * 10, 0), (10 * i, width))
    pygame.display.update()


def draw_blocks():
    for block in block_list:
        block.draw_block()


def draw_buttons():
    for button in button_list:
        button.draw()


def bubble_sort(draw_grid):
    sorted = False
    length = len(block_list)-1
    while not sorted:
        sorted = True
        for x in range(length):
            block_list[x].color = PURPLE
            if block_list[x].height > block_list[x+1].height:
                block_list[x].height, block_list[x+1].height = block_list[x+1].height, block_list[x].height
                block_list[x].y = width - block_list[x].height
                block_list[x+1].y = width - block_list[x+1].height
                sorted = False
            draw_grid()
            block_list[x].color = RED
        length += -1

def invert_sort(draw_grid):
    length = len(block_list) - 1
    for x in range(0, length):
        for i in range(x, length+1):
            block_list[x].color = PURPLE
            block_list[i].color = PURPLE
            if block_list[x].height > block_list[i].height:
                block_list[x].height, block_list[i].height = block_list[i].height, block_list[x].height
                block_list[x].y = width - block_list[x].height
                block_list[i].y = width - block_list[i].height
            draw_grid()
            block_list[x].color = RED
            block_list[i].color = RED


def main():
    for x in range(40):
        block = Block(x * 20)
        block.draw_block()
        block_list.append(block)

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