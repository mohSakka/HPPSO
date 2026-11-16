"""Additional non-separable, rotated and composition benchmarks.

These are explicitly defined local instances, not an official CEC/BBOB suite.
No duplicate Sphere/Rosenbrock cases or externally rescaled objectives.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import dct

VERSION = "additional-v3"
NAMES = ("f21_expanded_schaffer_f6", "f22_rotated_ellipsoid",
         "f23_rotated_rastrigin", "f24_rotated_ackley", "f25_composition",
         "f26_rotated_griewank", "f27_rotated_discus", "f28_schwefel_1_2",
         "f29_rotated_schaffer_f7", "f30_composition_multimodal")
BOUNDS = dict(zip(NAMES, ((-100., 100.), (-100., 100.), (-5.12, 5.12),
                          (-32.768, 32.768), (-5., 5.), (-600.,600.), (-100.,100.),
                          (-100.,100.), (-100.,100.), (-5.,5.))))


@dataclass
class AdditionalProblem:
    name: str
    dimension: int
    shift: np.ndarray
    permutations: np.ndarray
    signs: np.ndarray
    center_offsets: np.ndarray

    @property
    def bounds(self):
        lo, hi = BOUNDS[self.name]
        return self.shift + lo, self.shift + hi

    def rotate(self, z, component=0):
        return dct(self.signs[component] * z[self.permutations[component]],
                   type=2, norm="ortho")

    @staticmethod
    def ellipsoid(z):
        return float(np.dot(np.geomspace(1., 1e6, len(z)), z*z))

    @staticmethod
    def rastrigin(z):
        return float(np.sum(z*z + 20*np.sin(np.pi*z)**2))

    @staticmethod
    def ackley(z):
        # Algebraically equivalent to the usual formula; exactly zero at zero.
        return float(-20*np.expm1(-.2*np.sqrt(np.mean(z*z)))
                     - np.e*np.expm1(-2*np.mean(np.sin(np.pi*z)**2)))

    @staticmethod
    def expanded_schaffer(z):
        r2 = z*z + np.roll(z, -1)**2
        t = .001*r2
        # Equivalent to .5 + (sin(sqrt(r2))**2 - .5)/(1+t)**2.
        # This form avoids cancellation near the optimum.
        return float(np.sum((.5*t*(2+t) + np.sin(np.sqrt(r2))**2)/(1+t)**2))

    @staticmethod
    def griewank(z):
        return float(np.sum(z*z)/4000 + 1 - np.prod(np.cos(z/np.sqrt(np.arange(1,len(z)+1)))))

    @staticmethod
    def discus(z):
        return float(1e6*z[0]**2 + np.sum(z[1:]**2))

    @staticmethod
    def schwefel_1_2(z):
        return float(np.sum(np.cumsum(z)**2))

    @staticmethod
    def schaffer_f7(z):
        radii = np.hypot(z[:-1], z[1:])
        return float(np.mean(np.sqrt(radii)*(1+np.sin(50*radii**.2)**2))**2)

    def __call__(self, x):
        z = np.asarray(x, dtype=float) - self.shift
        if self.name == NAMES[0]:
            return self.expanded_schaffer(z)
        if self.name == NAMES[7]:
            return self.schwefel_1_2(z)
        if self.name not in (NAMES[4], NAMES[9]):
            y = self.rotate(z)
            return {NAMES[1]: self.ellipsoid, NAMES[2]: self.rastrigin,
                    NAMES[3]: self.ackley, NAMES[5]: self.griewank,
                    NAMES[6]: self.discus, NAMES[8]: self.schaffer_f7}[self.name](y)
        # Three distinct centers; spatially varying normalized weights.
        offsets = z - self.center_offsets
        distances = np.sum(offsets*offsets, axis=1)
        exact = np.flatnonzero(distances == 0)
        biases = np.array([0., 100., 200.])
        if exact.size:
            return float(biases[exact[0]])
        widths = np.array([1.,2.,3.]) if self.name == NAMES[9] else np.ones(3)
        log_weights = -.5*np.log(distances) - distances/(2*self.dimension*widths**2)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.sum(weights)
        components = (self.rastrigin, self.griewank, self.expanded_schaffer) if self.name == NAMES[9] else (self.ellipsoid, self.rastrigin, self.ackley)
        values = np.array([f(self.rotate(offsets[k],k)) for k,f in enumerate(components)])
        return float(weights @ (values + biases))

    def definition(self):
        lo, hi = self.bounds
        base_lo, base_hi = BOUNDS[self.name]
        rotation = None if self.name in (NAMES[0],NAMES[7]) else "DCT-II(signs[k] * z[permutations[k]]), norm=ortho"
        return dict(name=self.name, dimension=self.dimension, version=VERSION,
                    shift_vector=self.shift.tolist(), base_bounds=[[base_lo]*self.dimension,[base_hi]*self.dimension],
                    bounds=[lo.tolist(),hi.tolist()], optimum=self.shift.tolist(), optimum_value=0.,
                    objective_scale="raw", shift_convention="F(x)=f(x-o); x in [L+o,U+o]",
                    rotation=rotation, permutations=self.permutations.tolist(), signs=self.signs.tolist(),
                    center_offsets=self.center_offsets.tolist(),
                    composition_widths=([1.,2.,3.] if self.name==NAMES[9] else [1.,1.,1.]) if self.name in (NAMES[4],NAMES[9]) else None,
                    composition_components=(["rastrigin","griewank","expanded_schaffer_f6"] if self.name==NAMES[9] else ["ellipsoid","rastrigin","ackley"]) if self.name in (NAMES[4],NAMES[9]) else None,
                    composition_biases=[0.,100.,200.] if self.name in (NAMES[4],NAMES[9]) else None,
                    composition_weight="d_i^(-1/2)*exp(-d_i/(2D*sigma_i^2)), sigma_i=composition_widths, d_i=||x-o-c_i||^2; normalized; exact center selects its component" if self.name in (NAMES[4],NAMES[9]) else None)

    def as_problem(self):
        definition = self.definition()
        return dict(name=self.name, dimension=self.dimension, function=self, bounds=self.bounds,
                    base_bounds=definition["base_bounds"], shift_vector=self.shift.tolist(),
                    shift_convention=definition["shift_convention"], objective_scale="raw",
                    definition=definition)


def build_additional_suite(dimension, seed=42, names=NAMES, fixed_shifts=None):
    if not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError("Additional benchmarks require dimension >= 2")
    if not names or len(set(names)) != len(names) or set(names)-set(NAMES):
        raise ValueError("Unknown, empty or duplicate additional function selection")
    problems = []
    for name in names:
        rng = np.random.default_rng(np.random.SeedSequence([seed, dimension, 21+NAMES.index(name)]))
        lo, hi = BOUNDS[name]
        shift = rng.uniform(.8*lo, .8*hi, dimension)
        if fixed_shifts and name in fixed_shifts:
            shift = np.asarray(fixed_shifts[name], dtype=float).copy()
        if shift.shape != (dimension,) or not np.all(np.isfinite(shift)):
            raise ValueError("Shift must contain one finite value per dimension")
        n = 3 if name in (NAMES[4],NAMES[9]) else (0 if name in (NAMES[0],NAMES[7]) else 1)
        permutations = np.array([rng.permutation(dimension) for _ in range(n)], dtype=int).reshape(n, dimension)
        signs = rng.choice([-1.,1.], size=(n,dimension))
        centers = np.zeros((3,dimension)) if n == 3 else np.empty((0,dimension))
        if n == 3:
            centers[1:] = rng.uniform(-4.,4.,size=(2,dimension))
        problems.append(AdditionalProblem(name, dimension, shift, permutations, signs, centers))
    return problems
