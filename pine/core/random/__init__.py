from jax import random
from config import DEFAULT_PRNG_STATE


def get_key_from_seed(seed: int):
    key = random.PRNGKey(seed)
    return key


def ball(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.ball(key, *args, **kwargs)


def bernoulli(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.ball(key, *args, **kwargs)


def categorical(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.categorical(key, *args, **kwargs)


def cauchy(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.cauchy(key, *args, **kwargs)


def choice(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.choice(key, *args, **kwargs)


def dirichlet(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.dirichlet(key, *args, **kwargs)


def double_sided_maxwell(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.double_sided_maxwell(key, *args, **kwargs)


def exponential(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.exponential(key, *args, **kwargs)


def fold_in(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.fold_in(key, *args, **kwargs)


def gamma(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.gamma(key, *args, **kwargs)


def generalized_normal(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.generalized_normal(key, *args, **kwargs)


def gumbel(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.gumbel(key, *args, **kwargs)


def loggamma(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.loggamma(key, *args, **kwargs)


def logistic(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.logistic(key, *args, **kwargs)


def maxwell(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.maxwell(key, *args, **kwargs)


def multivariate_normal(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.multivariate_normal(key, *args, **kwargs)


def normal(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.normal(key, *args, **kwargs)


def orthogonal(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.orthogonal(key, *args, **kwargs)


def pareto(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.pareto(key, *args, **kwargs)


def permutation(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.permutation(key, *args, **kwargs)


def rademacher(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.rademacher(key, *args, **kwargs)


def poisson(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.poisson(key, *args, **kwargs)


def uniform(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.poisson(key, *args, **kwargs)


def laplace(seed: int = DEFAULT_PRNG_STATE, *args, **kwargs):
    key = get_key_from_seed(seed)
    return random.laplace(key, *args, **kwargs)
