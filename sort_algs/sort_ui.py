import pygame
import time
import sys, os
from sort_block import *
from sortbubble import *

global screen
global time1

screen = pygame.display.set_mode((500, 500))


def main():

    pygame.init()
    pygame.display.set_caption("SORT")
    all_sprites_list = pygame.sprite.Group()
    block_list = []
    all_sprites_list.empty()

    getNumbersInList()

    for x in range(len(listNumbers)-1):
        newblock = Block(listNumbers[x], x)
        all_sprites_list.add(newblock)
        block_list.append(newblock)

    # CLOCK
    clock = pygame.time.Clock()
    clock.tick(60)
    time1 = 0
    running = True

    while running:


        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    sort()
                if event.key == pygame.K_SPACE:
                    sortDiffrent()
                if event.key == pygame.K_r:
                    main()

        def sort():
            length = len(block_list) - 1
            sorted = False

            # while not sorted:
            #     tick = clock.tick()
            #     time1 += tick
            #     sorted = True
            #     for x in range(0, length):
            #         if list[x] > list[x + 1]:
            #             sorted = False
            #             if time1 > 10:
            #                 all_sprites_list.empty()
            #                 list[x], list[x + 1] = list[x + 1], list[x]
            #                 for x in range(len(list) - 1):
            #                     screen.fill('black')
            #
            #                     newblock = Block(list[x], x)
            #                     all_sprites_list.add(newblock)
            #
            #                     all_sprites_list.draw(screen)
            #                     pygame.display.update()
            #
            #                 time1 = 0
            while not sorted:
                sorted = True
                for x in range(0, length):
                    print(block_list[x].height)
                    if block_list[x].height > block_list[x + 1].height:
                        sorted = False

                        block_list[x].rect.left, block_list[x + 1].rect.left = block_list[x + 1].rect.left, block_list[x].rect.left
                        block_list[x], block_list[x+1] = block_list[x + 1], block_list[x]
                        screen.fill('black')
                        block_list[x+1].image.fill(PINK)
                        all_sprites_list.draw(screen)
                        pygame.display.update()

                    if not sorted:
                        block_list[x].image.fill(RED)
                    if sorted:
                        block_list[length - x].image.fill(GREEN)


        def sortDiffrent():
            length = len(block_list) - 1
            sorted = False
            while not sorted:
                sorted = True
                for x in range(0, length):
                    for i in range(x, length+1):
                        if block_list[x].height > block_list[i].height:
                            sorted = False

                            block_list[x].rect.left, block_list[i].rect.left = block_list[i].rect.left, block_list[x].rect.left
                            block_list[x], block_list[i] = block_list[i], block_list[x]
                            screen.fill('black')
                            block_list[i].image.fill(PINK)
                            all_sprites_list.draw(screen)
                            pygame.display.update()
                            #break an/aus egal

                        if not sorted:
                            block_list[x].image.fill(RED)
                        if sorted:
                            block_list[length - x].image.fill(GREEN)

        all_sprites_list.draw(screen)
        pygame.display.update()


main()


def sort_bubble(list):
    length = len(list) -1

    while not sorted:
        sorted = True
        for x in range(0,length):
            if list[x]>list[x+1]:
                list[x], list[x+1] = list[x+1], list[x]
                sorted = False
