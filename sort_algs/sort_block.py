import pygame
import random

GREEN = (0, 255, 0)
RED = (255, 0, 0)
PINK = (150, 75, 200)


class Block(pygame.sprite.Sprite):
    def __init__(self, height, index):
        super().__init__()
        self.height = height
        self.image = pygame.Surface([5, self.height])
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.left = 5*index
        self.rect.top = 0

    def updateHeight(self, newHeight):
        self.height = newHeight
        self.image = pygame.Surface([5, self.height])