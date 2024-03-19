import random

import numpy as np
import pygame
import tensorflow as tf


class Ai:
    def __init__(self, model=None, mutation_rate=0.01):
        """
        Initialize a GeneticModel instance.

        Parameters:
        - model: Pre-existing Keras model to use. If None, a new model will be randomly initialized.
        - mutation_rate: Rate of mutation to apply when creating a mutated model.
        """
        if model is None:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, input_shape=(4,), activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    initial_weights = layer.get_weights()
                    initial_weights[0] = np.random.randn(*initial_weights[0].shape)  # Random weights
                    initial_weights[1] = np.random.randn(*initial_weights[1].shape)  # Random biases
                    layer.set_weights(initial_weights)
        else:
            self.model = tf.keras.models.clone_model(model)
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    mutated_weights = layer.get_weights()
                    mutated_weights[0] += mutation_rate * np.random.randn(*mutated_weights[0].shape)
                    mutated_weights[1] += mutation_rate * np.random.randn(*mutated_weights[1].shape)
                    layer.set_weights(mutated_weights)

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def predict_jump(self, distance, velocity, distance_top, distance_bottom):
        input_data = np.array([[distance, velocity, distance_top, distance_bottom]])
        prediction = self.model.predict(input_data)
        return prediction > 0.5


class Game:
    def __init__(self, human=True, algorithm_random=0, algorithm=0, ai=0):
        pygame.init()
        self.width, self.height = 1000, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.hole = 175
        self.gap = 175
        self.running = True
        self.bars = np.array([random.randint(100, self.height - 100 - self.hole) for _ in range(15)])
        self.pos = 0
        self.result = False
        self.human = None
        self.klicked = False
        self.highscore = 0
        self.algorithm = np.array([])
        self.algorithm_random = np.array([])
        self.ai = []
        self.ai_values = np.array([])
        self.num_rndm = algorithm_random
        self.num_ai = ai
        if human:
            self.human = np.array([500, 0, 0])
        if algorithm > 0:
            self.algorithm = np.array([[500, 0, 0] for _ in range(algorithm)])
        if algorithm_random > 0:
            self.algorithm_random = np.array([[500, 0, 0] for _ in range(algorithm_random)])
        if ai > 0:
            self.ai = [Ai() for _ in range(ai)]
            self.ai_values = np.array([[500, 0, 0] for _ in range(ai)])
        self.top_ai = []

    def draw(self):
        self.screen.fill((0, 0, 0))
        for idx, i in enumerate(self.bars):
            pygame.draw.rect(self.screen, (255, 255, 255),
                             pygame.Rect(self.gap - self.pos + idx * self.gap + 100,
                                         0, 25, i))
            pygame.draw.rect(self.screen, (255, 255, 255),
                             pygame.Rect(self.gap - self.pos + idx * self.gap + 100,
                                         i + self.hole, 25, self.height - i - self.hole))
        for algorithm in self.algorithm:
            pygame.draw.circle(self.screen,
                               (255 - int(155 / (algorithm[2] + 1)), 255 - int(155 / (algorithm[2] + 1)), 100),
                               (150, self.height - algorithm[0]), 15)
        for algorithm in self.algorithm_random:
            pygame.draw.circle(self.screen,
                               (100, 255 - int(155 / (algorithm[2] + 1)), 255 - int(155 / (algorithm[2] + 1))),
                               (150, self.height - algorithm[0]), 15)
        for algorithm in self.ai_values:
            pygame.draw.circle(self.screen,
                               (255 - int(155 / (algorithm[2] + 1)), 100,  255 - int(155 / (algorithm[2] + 1))),
                               (150, self.height - algorithm[0]), 15)
        if self.human is not None:
            pygame.draw.circle(self.screen, (100, 255, 100), (150, self.height - self.human[0]), 15)
        font = pygame.font.Font(None, 24)
        high_score_text = f"Highscore: {self.highscore}"
        high_score_text_surface = font.render(high_score_text, True, (0, 0, 0))
        text_width = high_score_text_surface.get_width()
        y = 15
        if self.human is not None:
            y += 20
        if len(self.algorithm) > 0:
            y += 20
            max_val = max([a[2] for a in self.algorithm])
            score_text = f"Algorithm: {max_val}"
            algorithm_score_text_surface = font.render(score_text, True, (0, 0, 0))
        if len(self.algorithm_random) > 0:
            y += 20
            max_val = max([a[2] for a in self.algorithm_random])
            score_text = f"Random Algorithm: {max_val}"
            algorithm_random_score_text_surface = font.render(score_text, True, (0, 0, 0))
            text_width = max(algorithm_random_score_text_surface.get_width(), text_width)
        if len(self.ai_values) > 0:
            y += 20
            max_val = max([a[2] for a in self.ai_values])
            score_text = f"Ai: {max_val}"
            ai_score_text_surface = font.render(score_text, True, (0, 0, 0))

        pygame.draw.rect(self.screen, (255, 255, 255),
                         pygame.Rect(10, 10, max(100, text_width + 10), y + 11))

        y = 15
        self.screen.blit(high_score_text_surface, (15, y))
        if self.human is not None:
            y += 20
            score_text = f"Score: {self.human[2]}"
            human_score_text_surface = font.render(score_text, True, (0, 0, 0))
            self.screen.blit(human_score_text_surface, (15, y))
        if len(self.algorithm) > 0:
            y += 20
            self.screen.blit(algorithm_score_text_surface, (15, y))
        if len(self.algorithm_random) > 0:
            y += 20
            self.screen.blit(algorithm_random_score_text_surface, (15, y))
        if len(self.ai_values) > 0:
            y += 20
            self.screen.blit(ai_score_text_surface, (15, y))
        pygame.display.flip()

    def check_collision(self, ball):
        if not 0 < ball < self.height:
            return True
        if self.gap - self.pos - 65 < 0 < self.gap - self.pos - 9:
            if not self.bars[0] + 15 < self.height - ball < self.bars[0] + self.hole:
                return True
        return False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.klicked = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.klicked = True

    def draw_result(self):
        pygame.draw.rect(self.screen, (255, 255, 255),
                         pygame.Rect(self.width / 2 - 250, self.height / 2 - 100, 500, 200))

        score_text = f"Score: {self.human[2]}"

        score_text_surface = pygame.font.Font(None, 36).render(score_text, True, (0, 0, 0))
        text_width, text_height = score_text_surface.get_size()
        self.screen.blit(score_text_surface, (self.width / 2 - text_width / 2, self.height / 2 - text_height / 2))
        pygame.display.flip()

    def algorithmic_move(self, ball):
        if self.bars[0] + self.hole - 50 < self.height - ball:
            return True
        return False

    def algorithmic_random(self, ball):
        if random.random() > 0.95:
            return random.getrandbits(1)
        if self.bars[0] + self.hole - 50 < self.height - ball:
            if random.random() > 0.95:
                return random.getrandbits(1)
            return True
        return False

    def ai_move(self, pos):
        distance = self.gap - self.pos
        velocity = self.ai_values[pos][1]
        distance_top = self.height - self.ai_values[pos][0] - self.bars[0] - 15
        distance_bottom = -(self.height - self.ai_values[pos][0] - self.bars[0] + 15 - self.hole)
        return self.ai[pos].predict_jump(distance, velocity, distance_top, distance_bottom)

    def run(self):
        while self.running:
            self.handle_events()
            if self.klicked:
                if self.human is not None:
                    self.human[1] = 10
                self.klicked = False
            self.draw()
            self.pos += 2

            # handles human controlled player
            if self.human is not None:
                self.human[0] += self.human[1]
                self.human[1] -= 1
                if self.check_collision(self.human[0]):
                    self.human = np.array([self.height - self.bars[0] - 50, 0, 0])

            # handles (perfect) algorithms
            idx_to_delete = []
            for idx, algorithm in enumerate(self.algorithm):
                algorithm[0] += algorithm[1]
                algorithm[1] -= 1
                if self.algorithmic_move(algorithm[0]):
                    algorithm[1] = 10
                if self.check_collision(algorithm[0]):  # kinda useless bc makes no mistakes...
                    idx_to_delete.append(idx)
            self.algorithm = np.delete(self.algorithm, idx_to_delete, axis=0)

            # handles algorithms with some random moves
            idx_to_delete = []
            for idx, algorithm in enumerate(self.algorithm_random):
                algorithm[0] += algorithm[1]
                algorithm[1] -= 1
                if self.algorithmic_random(algorithm[0]):
                    algorithm[1] = 10
                if self.check_collision(algorithm[0]):
                    idx_to_delete.append(idx)
            self.algorithm_random = np.delete(self.algorithm_random, idx_to_delete, axis=0)
            while len(self.algorithm_random) < self.num_rndm:
                row_to_append = np.array([[self.height - self.bars[0] - 50, 0, 0]])
                self.algorithm_random = np.append(self.algorithm_random, row_to_append, axis=0)

            # handles AI controlled algorithms
            idx_to_delete = []
            for idx, algorithm in enumerate(self.ai_values):
                algorithm[0] += algorithm[1]
                algorithm[1] -= 1
                if self.ai_move(idx):
                    algorithm[1] = 10
                if self.check_collision(algorithm[0]):
                    idx_to_delete.append(idx)

            self.ai_values = np.delete(self.ai_values, idx_to_delete, axis=0)
            new_ai_list = []
            for idx, ai in enumerate(self.ai):
                if idx not in idx_to_delete:
                    new_ai_list.append(ai)
            self.ai = new_ai_list

            self.top_ai.extend(self.ai)
            if len(self.top_ai) > self.num_ai:
                self.top_ai = self.top_ai[-self.num_ai:]

            if len(self.ai_values) == 0 and self.num_ai > 0:
                self.ai = [Ai(ai.model) for ai in self.top_ai]
                while len(self.ai) < self.num_ai:
                    self.ai.append(Ai())
                self.ai_values = np.array([[self.height - self.bars[0] - 50, 0, 0] for _ in range(self.num_ai)])

            # add points if new hole is passed
            if self.pos >= self.gap:
                if self.human is not None:
                    self.human[2] += 1
                    self.highscore = max(self.highscore, self.human[2])
                for algorithm in self.algorithm:
                    algorithm[2] += 1
                    self.highscore = max(self.highscore, algorithm[2])
                for algorithm in self.algorithm_random:
                    algorithm[2] += 1
                    self.highscore = max(self.highscore, algorithm[2])
                for algorithm in self.ai_values:
                    algorithm[2] += 1
                    self.highscore = max(self.highscore, algorithm[2])
                self.pos = 0
                self.bars = np.delete(self.bars, 0)
                self.bars = np.append(self.bars, random.randint(100, self.height - 100 - self.hole))
            pygame.time.delay(25)
            if self.pos == 10:
                break


if __name__ == '__main__':
    game = Game(human=False, algorithm=0, algorithm_random=0, ai=10)
    game.run()
