"""Function registry for the extended trait study; old benchmark instances preserved."""
from functools import lru_cache
import numpy as np
from hppso.benchmarks.additional import AdditionalProblem
from experiments.final_benchmark import suite as original_suite

SUPPORTED_FUNCTIONS = (
    "f1_sphere", "f2_schwefel_2_22", "f3_zakharov", "f4_rosenbrock",
    "f5_powell_sum", "f6_rastrigin", "f7_ackley",
    "f8_griewank", "f9_weierstrass", "f10_levy", "f11_schwefel_2_26",
    "f13_michalewicz", "f14_alpine_1", "f17_bent_cigar", "f18_discus",
    "f19_penalized_1", "f21_expanded_schaffer_f6", "f22_rotated_ellipsoid",
    "f23_rotated_rastrigin", "f24_rotated_ackley", "f25_composition",
    "f26_rotated_griewank", "f27_rotated_discus", "f28_schwefel_1_2", "f29_rotated_schaffer_f7",
    "f30_composition_multimodal", "f31_composition_ridges",
)


class RidgeComposition(AdditionalProblem):
    """Three independently rotated Rosenbrock/discus/Schaffer-F7 components."""
    @property
    def bounds(self):return self.shift-5.,self.shift+5.

    @staticmethod
    def centered_rosenbrock(z):
        u=z+1
        return float(np.sum(100*(u[1:]-u[:-1]**2)**2+(1-u[:-1])**2))

    def __call__(self,x):
        offsets=np.asarray(x,dtype=float)-self.shift-self.center_offsets
        distance=np.sum(offsets*offsets,axis=1)
        biases=np.array([0.,100.,200.]);widths=np.array([1.,2.,3.])
        exact=np.flatnonzero(distance==0)
        if exact.size:return float(biases[exact[0]])
        logs=-.5*np.log(distance)-distance/(2*self.dimension*widths**2)
        weights=np.exp(logs-logs.max());weights/=weights.sum()
        functions=(self.centered_rosenbrock,self.discus,self.schaffer_f7)
        values=np.array([f(self.rotate(offsets[k],k)) for k,f in enumerate(functions)])
        return float(weights@(values+biases))

    def definition(self):
        lo,hi=self.bounds
        return dict(name=self.name,dimension=self.dimension,version="trait-ridge-composition-v1",
            shift_vector=self.shift.tolist(),base_bounds=[[-5.]*self.dimension,[5.]*self.dimension],
            bounds=[lo.tolist(),hi.tolist()],optimum=self.shift.tolist(),optimum_value=0.,
            objective_scale="raw",shift_convention="F(x)=f(x-o); x in [L+o,U+o]",
            rotation="DCT-II(signs[k]*z[permutations[k]]), norm=ortho",
            permutations=self.permutations.tolist(),signs=self.signs.tolist(),
            center_offsets=self.center_offsets.tolist(),composition_widths=[1.,2.,3.],
            composition_biases=[0.,100.,200.],
            composition_components=["rosenbrock(y+1)","discus","schaffer_f7"],
            composition_weight="d_i^(-1/2)*exp(-d_i/(2D*sigma_i^2)); normalized; exact center selects its component")


@lru_cache(maxsize=8)
def trait_suite(dimension,seed):
    problems=dict(original_suite(dimension,seed))
    rng=np.random.default_rng(np.random.SeedSequence([seed,dimension,31]))
    shift=rng.uniform(-4.,4.,dimension)
    permutations=np.array([rng.permutation(dimension) for _ in range(3)])
    signs=rng.choice([-1.,1.],size=(3,dimension))
    centers=np.zeros((3,dimension));centers[1:]=rng.uniform(-4.,4.,size=(2,dimension))
    problem=RidgeComposition("f31_composition_ridges",dimension,shift,permutations,signs,centers)
    problems[problem.name]=problem.as_problem()
    return problems
