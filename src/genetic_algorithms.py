import cv2
import random
import numpy as np
from PIL import Image
from utils.evaluate_function import HelperEvaluator
import os
from snowflake.snowflake import Snowflake


class GA:
    RANDOM_GENOME_SCALE: float = 7.0
    MUTATION_RATE: float = .2
    MUTATION_SCALE: float = .3
    CONVERGENCE_THRESHOLD: float = .03
    CONVERGENCE_WINDOW: int = 10

    """
    In this genetic algorithm implementation genome is a set of vertices, starting with the first one and ending with the last one
    """

    def __init__(self, verbose=True, genome_size=4, population_size=100,
                 evaluator: HelperEvaluator = None, img_format: str = 'jpg'):
        self.verbose = verbose
        self.GENOME_SIZE = genome_size
        self.population_size = population_size
        self.evaluator = evaluator
        self.img_format = img_format
        self.img_dir = './img_dir/'

        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def _random_genome(self):
        return np.random.rand(self.GENOME_SIZE) * self.RANDOM_GENOME_SCALE

    def _mutate(self, genome):
        mutation = np.random.normal(0, self.MUTATION_SCALE, self.GENOME_SIZE)
        return genome + mutation

    def mutate(self, genome, probability=.2):
        if random.random() <= self.MUTATION_RATE:
            return self._mutate(genome)
        return genome

    def crossover(self, genome1, genome2):
        crossover_point = random.randint(1, self.GENOME_SIZE - 1)
        p1 = genome1[:crossover_point]
        p2 = genome2[crossover_point:]
        return np.concatenate([p1, p2])

    def _render_image(self, genome, filename) -> None:
        s = Snowflake(genome)
        s.generate()
        final_img = s.draw(max_height=1000)
        img = Image.fromarray(final_img, 'RGB')
        img.save(filename)

    def render_population(self, population, epoch):
        epoch_dir = self._get_epoch_dir(epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        for genome_id, genome in enumerate(population):
            filename = self._get_genome_image_name(genome_id)
            filename = os.path.join(epoch_dir, filename)
            self._render_image(genome, filename=filename)

    def _get_epoch_dir(self, epoch: int) -> str:
        directory = f'epoch_{epoch}'
        return os.path.join(self.img_dir, directory)

    def _get_genome_image_name(self, genome_id: int) -> str:
        return f'{genome_id}.jpg'

    def get_fitness(self, epoch):
        directory = self._get_epoch_dir(epoch)
        scores = self.evaluator.evaluate(image_dir=directory, img_format=self.img_format)
        return list(map(lambda x: x['mean_score_prediction'], scores))

    def run(self, num_epochs, top_percent=.4):
        history = []
        population = [self._random_genome() for i in range(self.population_size)]
        epoch = 0
        while True:
            self.render_population(population=population, epoch=epoch)
            fitness = self.get_fitness(epoch=epoch)
            topk = np.argsort(fitness)[::-1][:int(top_percent * self.population_size)]

            reproduction_group = [population[i] for i in topk]
            random.shuffle(reproduction_group)

            children = []
            for i in range(self.population_size - len(reproduction_group)):
                p1, p2 = random.randint(0, len(reproduction_group) - 1), random.randint(0, len(reproduction_group) - 1)
                children.append(self.crossover(reproduction_group[p1], reproduction_group[p2]))

            population = reproduction_group + children
            population = [self.mutate(x) for x in population]

            max_fitness = np.max(fitness)

            if self.verbose:
                print(f'Epoch: {epoch}, Max fitness: {max_fitness}')
            history.append(max_fitness)

            if epoch > self.CONVERGENCE_WINDOW:
                window_begin = history[-self.CONVERGENCE_WINDOW]
                window_end = history[-1]
                diff = window_end - window_begin
                if self.verbose:
                    print(f'Window diff: {diff}')

                if diff < self.CONVERGENCE_THRESHOLD:
                    break
            epoch += 1
        self.render_population(population=population, epoch=epoch)
        fitness = self.get_fitness(epoch=epoch)
        return population[np.argmax(fitness)]


if __name__ == '__main__':
    WORKDIR: str = os.getcwd()
    BASE_MODEL_NAME: str = 'MobileNet'
    WEIGHTS_FILE: str = './weights.hdf5'

    he = HelperEvaluator(base_model_name=BASE_MODEL_NAME, weights_file=WEIGHTS_FILE)
    genetic_algorithms = GA(evaluator=he, population_size=100)
    genetic_algorithms.run(5)
