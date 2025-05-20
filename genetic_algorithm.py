from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances_argmin


class GAClustering:
    """
    Genetic Algorithm-based clustering model.

    Attributes
    ----------
        _clusters : int
            Number of clusters (must be ≥ 2).
        _pop_size : int, default=100
            Number of chromosomes in the population (must be > 0).
        _max_gen : int, default=100
            Maximum number of generations (must be > 0).
        _cross_rate : float, default=0.8
            Probability of crossover (must be in (0, 1)).
        _mut_rate : float, default=0.001
            Probability of mutation (must be in (0, 1)).
        _random_state : Optional[int], default=None
            Random seed
        _best_solution : Optional[NDArray], default=None
            Solution with the highest fitness
        _best_fitness : float, default=-np.inf
            The highest fitness
        _centers : Optional[NDArray], default=None
            Best centers
        _features : Optional[int], default=None
            Number of features in input data
        _population : Optional[List[NDArray]], default=None
            Population in the current generation

    Methods
    -------
        _init_population(X)
            Initialize population
        fit(X)
            Fit model to data
        predict(X)
            Assign cluster labels
        _decode(chromosome)
            Get centers from chromosome
        _fitness(chromosome, X)
            Compute fitness of the chromosome
        _selection(pop)
            Select chromosomes for the next generation
        _crossover(parent1, parent2)
            Crossover between two parents
        _mutate(chromosome)
            Mutate chromosome
    """

    def __init__(
        self,
        clusters: int,
        pop_size: int = 100,
        max_gen: int = 100,
        cross_rate: float = 0.8,
        mut_rate: float = 0.001,
        random_state: Optional[int] = None,
    ):
        """
        Set attributes.

        Raises
        ------
            ValueError
                If any parameter is out of valid range

        Parameters
        ----------
        clusters : int
            Number of clusters (must be ≥ 2).
        pop_size : int, default=100
            Number of chromosomes in the population (must be > 0).
        max_gen : int, default=100
            Maximum number of generations (must be > 0).
        cross_rate : float, default=0.8
            Probability of crossover (must be in (0, 1)).
        mut_rate : float, default=0.001
            Probability of mutation (must be in (0, 1)).
        random_state : Optional[int], default=None
            Random seed for reproducibility.

        """
        if clusters < 2:
            raise ValueError("`clusters` must be at least 2")
        if pop_size <= 0:
            raise ValueError("`pop_size` must be positive")
        if max_gen <= 0:
            raise ValueError("`max_gen` must be positive")
        if not (0 < cross_rate < 1):
            raise ValueError("`cross_rate` must be in (0, 1)")
        if not (0 < mut_rate < 1):
            raise ValueError("`mut_rate` must be in (0, 1)")

        self._clusters: int = clusters
        self._pop_size: int = pop_size
        self._max_gen: int = max_gen
        self._cross_rate: float = cross_rate
        self._mut_rate: float = mut_rate
        self._random_state: Optional[int] = random_state
        self._best_solution: Optional[NDArray] = None
        self._best_fitness: float = -np.inf
        self._centers: Optional[NDArray] = None
        self._features: Optional[int] = None
        self._population: Optional[List[NDArray]] = None

    def _init_population(self, X: NDArray) -> List[NDArray]:
        """
        Initialize population with randomly chosen samples from the input dataset.

        Parameters
        ----------
        X : NDArray
            Input dataset

        Returns
        -------
        population : List[NDArray]
            List of flattened chomosomes
        """
        return [
            X[
                np.random.choice(len(X), self._clusters, replace=False)
            ].flatten()
            for _ in range(self._pop_size)
        ]

    def fit(self, X: NDArray) -> None:
        """
        Fit the GA clustering algorithm to data.

        Parameters
        ----------
            X : NDArray
                Input data

        Returns
        -------
        None
        """
        if self._random_state is not None:
            np.random.seed(self._random_state)

        self._features = X.shape[1]
        self._population = self._init_population(X)

        for _ in range(self._max_gen):
            fitnesses = np.array(
                [self._fitness(chromo, X) for chromo in self._population]
            )
            elite_idx = np.argmax(fitnesses)

            if fitnesses[elite_idx] > self._best_fitness:
                self._best_fitness = fitnesses[elite_idx]
                self._best_solution = self._population[elite_idx].copy()

            new_population = (
                [self._best_solution.copy()]
                if self._best_solution is not None
                else []
            )
            selected = self._selection(self._population, fitnesses)

            while len(new_population) < self._pop_size:
                p1_idx, p2_idx = np.random.choice(
                    len(selected), 2, replace=False
                )
                c1, c2 = self._crossover(selected[p1_idx], selected[p2_idx])
                new_population.append(self._mutate(c1))
                if len(new_population) < self._pop_size:
                    new_population.append(self._mutate(c2))

            self._population = new_population

        if self._best_solution is not None:
            self._centers = self._decode(self._best_solution)

    def predict(self, X: NDArray) -> NDArray:
        """
        Assign cluster labels to each sample in the input data.

        Parameters
        ----------
            X : NDArray
                Input data to cluster

        Returns
        -------
            labels : NDArray
                Cluster indices for each sample

        Raises
        -------
            RuntimeError
                If called before the model has been fitted
            ValueError
                If invalid shape of the input data
        Returns
        -------
        labels : NDArray
            Predicted cluster labels
        """
        if self._centers is None or self._features is None:
            raise RuntimeError(
                "Model must be fitted before calling predict()."
            )
        if X.shape[1] != self._features:
            raise ValueError("Invalid shape of the input data.")
        return pairwise_distances_argmin(X, self._centers)

    def _decode(self, chromosome: NDArray) -> NDArray:
        """
        Get centers of clusters from the chromosome.

        Raises
        ------
        ValueError
            If attribute '_features' is not set

        Returns
        -------
        centers : NDArray
            Array of shape (clusters, features).
        """
        if self._features is None:
            raise ValueError(
                "Attribute '_features' must be set before calling _decode()."
            )
        return chromosome.reshape((self._clusters, self._features))

    def _fitness(self, chromosome: NDArray, X: NDArray) -> float:
        """
        Compute fitness value of solution.

        Parameters
        ----------
        chromosome : NDArray
            Array of cluster centers
        X : NDArray
            Input data to cluster
        """
        centers = self._decode(chromosome)
        labels = pairwise_distances_argmin(X, centers)
        dists = np.linalg.norm(X - centers[labels], axis=1)
        return 1.0 / float(np.sum(dists))

    def _selection(
        self, pop: List[NDArray], fitnesses: NDArray
    ) -> List[NDArray]:
        """
        Select chromosomes for the next generation using roulette wheel selection.

        Parameters
        ----------
        pop : List[NDArray]
            Current population
        fitnesses : NDArray
            Fitness scores for each chromosome

        Returns
        -------
        selected : List[NDArray]
            Selected chromosomes for mating
        """
        probs = fitnesses / np.sum(fitnesses)
        indices = np.random.choice(len(pop), size=len(pop), p=probs)
        return [pop[i].copy() for i in indices]

    def _crossover(
        self, parent1: NDArray, parent2: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        One-point crossover between two parents with defined crossover probability.

        Parameters
        ----------
        parent1, parent2 : NDArray
            Parents for crossover

        Returns
        -------
        children : Tuple[NDArray, NDArray]
            Two chromosomes
        """
        if np.random.rand() > self._cross_rate:
            return parent1.copy(), parent2.copy()
        point = np.random.randint(1, len(parent1))
        return (
            np.concatenate([parent1[:point], parent2[point:]]),
            np.concatenate([parent2[:point], parent1[point:]]),
        )

    def _mutate(self, chromosome: NDArray) -> NDArray:
        """
        Apply mutation to the chromosome with defined mutation probability.

        Parameters
        ----------
        chromosome : NDArray
            Сhromosome to mutate

        Returns
        -------
        chromosome : NDArray
            Mutated chromosome
        """
        for i in range(len(chromosome)):
            if np.random.rand() < self._mut_rate:
                d = np.random.rand()
                sign = 1 if np.random.rand() < 0.5 else -1
                chromosome[i] += (
                    sign * 2 * d * chromosome[i]
                    if chromosome[i] != 0
                    else sign * 2 * d
                )
        return chromosome

    @property
    def centers(self) -> NDArray:
        if self._centers is None:
            raise ValueError("Model must be fitted before getting centers.")
        return self._centers
