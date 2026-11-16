"""Vectorized independent variant paths and row-wise benchmark objectives."""
import numpy as np
from time import monotonic
from scipy.fft import dct
from hppso.algorithms._pso_bounds import reflect_bounds, velocity_limit


class BatchFunction:
    """Algebraically equivalent row-wise formulas; final axis is dimension.

    BLAS/reduction order may differ from scalar legacy evaluators. These studies
    are rerun consistently with this evaluator, without importing older scores.
    """
    def __init__(self, problem):
        self.name = problem["name"]
        self.shift = np.asarray(problem["shift_vector"])
        self.d = len(self.shift)
        self.extra = problem["function"]
        self.weights = np.geomspace(1., 1e6, self.d)
        if self.name == "f9_weierstrass":
            harmonics = np.arange(21)
            self.weierstrass_weights = 0.5**harmonics
            self.weierstrass_frequencies = 2 * np.pi * 3.0**harmonics
            self.weierstrass_offset = self.d * np.sum(
                self.weierstrass_weights * np.cos(0.5 * self.weierstrass_frequencies)
            )

    def rotate(self, z, component=0):
        p = self.extra
        return dct(np.ascontiguousarray(p.signs[component] * z[..., p.permutations[component]]),
                   type=2, norm="ortho", axis=-1, workers=1)

    def ellipsoid(self,z):return np.sum(self.weights*z*z,axis=-1)

    @staticmethod
    def rastrigin(z):return np.sum(z*z+20*np.sin(np.pi*z)**2,axis=-1)

    @staticmethod
    def ackley(z):
        return -20*np.expm1(-.2*np.sqrt(np.mean(z*z,axis=-1))) - np.e*np.expm1(-2*np.mean(np.sin(np.pi*z)**2,axis=-1))

    @staticmethod
    def griewank(z):
        return np.sum(z*z,axis=-1)/4000-np.prod(np.cos(z/np.sqrt(np.arange(1,z.shape[-1]+1))),axis=-1)+1

    @staticmethod
    def discus(z):return 1e6*z[...,0]**2+np.sum(z[...,1:]**2,axis=-1)

    @staticmethod
    def schaffer_f7(z):
        r=np.hypot(z[...,:-1],z[...,1:])
        return np.mean(np.sqrt(r)*(1+np.sin(50*r**.2)**2),axis=-1)**2

    @staticmethod
    def expanded_schaffer(z):
        r2=z*z+np.roll(z,-1,axis=-1)**2;t=.001*r2
        return np.sum((.5*t*(2+t)+np.sin(np.sqrt(r2))**2)/(1+t)**2,axis=-1)

    @staticmethod
    def centered_rosenbrock(z):
        u=z+1
        return np.sum(100*(u[...,1:]-u[...,:-1]**2)**2+(1-u[...,:-1])**2,axis=-1)

    def composition(self,z,ridges=False):
        offsets=z[...,None,:]-self.extra.center_offsets
        distance=np.sum(offsets*offsets,axis=-1);exact=distance==0;safe=np.where(exact,1.,distance)
        widths=np.array([1.,2.,3.])
        logs=-.5*np.log(safe)-safe/(2*self.d*widths**2)
        weights=np.exp(logs-logs.max(axis=-1,keepdims=True));weights/=weights.sum(axis=-1,keepdims=True)
        weights=np.where(exact.any(axis=-1)[...,None],np.eye(3)[exact.argmax(axis=-1)],weights)
        functions=(self.centered_rosenbrock,self.discus,self.schaffer_f7) if ridges else (self.rastrigin,self.griewank,self.expanded_schaffer)
        values=np.stack([f(self.rotate(offsets[...,k,:],k)) for k,f in enumerate(functions)],axis=-1)
        return np.sum(weights*(values+np.array([0.,100.,200.])),axis=-1)

    def __call__(self, x):
        z = np.asarray(x,dtype=float)-self.shift
        name=self.name
        if name=="f1_sphere":return np.sum(np.clip(z**2,0,1e300),axis=-1)
        if name == "f2_schwefel_2_22":
            absolute = np.abs(z)
            with np.errstate(divide="ignore"):
                log_product = np.sum(np.log(absolute), axis=-1)
            return np.logaddexp(np.log1p(np.sum(absolute, axis=-1)), log_product)
        if name == "f5_powell_sum":
            exponents = np.arange(2, self.d + 2) / self.d
            return np.sum(np.clip(np.abs(z), 0, 1e100)**exponents, axis=-1)
        if name=="f3_zakharov":
            a=np.sum(np.clip(z*z,0,1e300),axis=-1)
            b=np.clip(np.sum(.5*np.arange(1,self.d+1)*z,axis=-1),-1e75,1e75)
            return a+b*b+b**4
        if name=="f6_rastrigin":return 10*self.d+np.sum(np.clip(z*z,0,1e300)-10*np.cos(2*np.pi*z),axis=-1)
        if name=="f8_griewank":return self.griewank(z)
        if name == "f9_weierstrass":
            rows = z.reshape(-1, self.d)
            scores = np.empty(len(rows))
            # Bound temporary memory for high-dimensional populations.
            rows_per_block = max(1, 1_000_000 // (len(self.weierstrass_weights) * self.d))
            for start in range(0, len(rows), rows_per_block):
                block = slice(start, start + rows_per_block)
                phases = (rows[block, None, :] + 0.5) * self.weierstrass_frequencies[:, None]
                np.cos(phases, out=phases)
                phases *= self.weierstrass_weights[:, None]
                scores[block] = np.sum(phases, axis=(-2, -1)) - self.weierstrass_offset
            return scores.reshape(z.shape[:-1])
        if name == "f11_schwefel_2_26":
            return 418.9829 * self.d - np.sum(z * np.sin(np.sqrt(np.abs(z))), axis=-1)
        if name=="f13_michalewicz":
            phase = np.arange(1, self.d + 1) * z**2 / np.pi
            return -np.sum(np.sin(z) * np.sin(phase)**20, axis=-1)
        if name=="f14_alpine_1":return np.sum(np.abs(z*np.sin(z)+.1*z),axis=-1)
        if name=="f17_bent_cigar":return z[...,0]**2+1e6*np.sum(np.clip(z[...,1:]**2,0,1e294),axis=-1)
        if name=="f18_discus":return 1e6*z[...,0]**2+np.sum(np.clip(z[...,1:]**2,0,1e300),axis=-1)
        if name=="f19_penalized_1":
            y=1+(z+1)/4
            penalty=np.sum(100*np.clip(np.abs(z)-10,0,1e75)**4,axis=-1)
            return (np.pi/self.d)*(10*np.sin(np.pi*y[...,0])**2+(y[...,-1]-1)**2
                +np.sum((y[...,:-1]-1)**2*(1+10*np.sin(np.pi*y[...,1:])**2),axis=-1))+penalty
        if name=="f4_rosenbrock":
            sq=np.clip(z[...,:-1]**2,-1e150,1e150)
            return np.sum(np.clip(100*(z[...,1:]-sq)**2+(1-z[...,:-1])**2,0,1e300),axis=-1)
        if name=="f10_levy":
            w=1+(z-1)/4
            return (np.sin(np.pi*w[...,0])**2+(w[...,-1]-1)**2*(1+np.sin(2*np.pi*w[...,-1])**2)
                +np.sum((w[...,:-1]-1)**2*(1+10*np.sin(np.pi*w[...,:-1]+1)**2),axis=-1))
        if name=="f7_ackley":
            return (-20*np.exp(-.2*np.sqrt(np.sum(np.clip(z*z,0,1e300),axis=-1)/self.d))
                -np.exp(np.sum(np.cos(2*np.pi*z),axis=-1)/self.d)+20+np.e)
        if name=="f21_expanded_schaffer_f6":
            r2=z*z+np.roll(z,-1,axis=-1)**2;t=.001*r2
            return np.sum((.5*t*(2+t)+np.sin(np.sqrt(r2))**2)/(1+t)**2,axis=-1)
        if name=="f22_rotated_ellipsoid":return self.ellipsoid(self.rotate(z))
        if name=="f23_rotated_rastrigin":return self.rastrigin(self.rotate(z))
        if name=="f24_rotated_ackley":return self.ackley(self.rotate(z))
        if name=="f26_rotated_griewank":return self.griewank(self.rotate(z))
        if name=="f27_rotated_discus":return self.discus(self.rotate(z))
        if name=="f28_schwefel_1_2":return np.sum(np.cumsum(z,axis=-1)**2,axis=-1)
        if name=="f29_rotated_schaffer_f7":return self.schaffer_f7(self.rotate(z))
        if name=="f30_composition_multimodal":return self.composition(z)
        if name=="f31_composition_ridges":return self.composition(z,True)
        if name=="f25_composition":
            offsets=z[...,None,:]-self.extra.center_offsets
            distances=np.sum(offsets*offsets,axis=-1)
            exact=distances==0
            safe=np.where(exact,1.,distances)
            logs=-.5*np.log(safe)-safe/(2*self.d)
            weights=np.exp(logs-np.max(logs,axis=-1,keepdims=True))
            weights/=np.sum(weights,axis=-1,keepdims=True)
            selected=np.eye(3)[exact.argmax(axis=-1)]
            weights=np.where(exact.any(axis=-1)[...,None],selected,weights)
            values=np.stack([self.ellipsoid(self.rotate(offsets[...,0,:],0)),
                self.rastrigin(self.rotate(offsets[...,1,:],1)),self.ackley(self.rotate(offsets[...,2,:],2))],axis=-1)
            return np.sum(weights*(values+np.array([0.,100.,200.])),axis=-1)
        raise ValueError(f"Unsupported batched benchmark: {name}")


class BatchBudget:
    def __init__(self,function,budget,paths,dimension,points,progress=None):
        self.function,self.budget=function,budget
        self.count=0
        self.best=np.full(paths,np.inf)
        self.best_positions=np.zeros((paths,dimension))
        n=min(points,budget)
        self.checkpoints=np.unique(np.r_[np.geomspace(1,budget,n).astype(int),np.linspace(1,budget,n).astype(int)])
        self.trace=[]
        self.progress = progress
        self.next_progress = 0.0

    def evaluate(self,positions):
        n=min(positions.shape[1],self.budget-self.count)
        if n<1:raise RuntimeError("Attempted evaluation beyond budget")
        values=self.function(positions[:,:n,:])
        self.count+=n
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite batch objective: "+str(np.argwhere(~np.isfinite(values))[:10].tolist()))
        cumulative=np.minimum.accumulate(values,axis=1)
        cumulative=np.minimum(cumulative,self.best[:,None])
        begin=self.count-n
        checkpoints=self.checkpoints[(self.checkpoints>begin)&(self.checkpoints<=self.count)]
        for checkpoint in checkpoints:
            self.trace.append((int(checkpoint),cumulative[:,checkpoint-begin-1].copy()))
        indices=values.argmin(axis=1);rows=np.arange(len(self.best))
        improved=values[rows,indices]<self.best
        self.best_positions[improved]=positions[rows[improved],indices[improved]]
        self.best=np.minimum(self.best,cumulative[:,-1])
        if self.progress is not None:
            now = monotonic()
            if now >= self.next_progress or self.count == self.budget:
                self.progress(self.count)
                self.next_progress = now + 5
        return values


class BatchedTraitPSO:
    """State shape (paths, particles, dimensions); no cross-path attraction."""
    def __init__(self,tracker,bounds,population,seed,specs,settings):
        self.tracker,self.settings=tracker,settings
        self.specs=specs;self.k=len(specs);self.n=population
        self.lower,self.upper=np.asarray(bounds,dtype=float);self.d=len(self.lower)
        self.width=self.upper-self.lower;self.limit=velocity_limit(self.lower,self.upper)
        self.rng=np.random.RandomState(seed);self.hp_rng=np.random.RandomState(seed)
        self.streams=[np.random.default_rng(np.random.SeedSequence([seed,900+i])) for i in range(7)]
        initial=self.rng.uniform(self.lower,self.upper,(population,self.d))
        self.hp_rng.uniform(self.lower,self.upper,(population,self.d))
        initial_v=np.zeros_like(initial)
        self.hp=np.array([s["backbone"]=="hp" for s in specs])
        self.positions=np.broadcast_to(initial,(self.k,population,self.d)).copy()
        self.velocities=np.where(self.hp[:,None,None],0,np.broadcast_to(initial_v,self.positions.shape)).copy()
        traits=self.streams[0].random((population,4))
        self.traits=np.broadcast_to(traits,(self.k,population,4)).copy()
        shared=traits.mean(axis=0) if settings["shared_level"]=="initial_mean" else np.full(4,settings["shared_level"])
        self.shared=np.array([s.get("personalization")=="shared" for s in specs])
        self.traits[self.shared]=shared
        self.q=np.array(["curiosity" in s["traits"] for s in specs])
        self.uniform_curiosity = self.q & np.array([s.get("personalization") == "uniform" for s in specs])
        self.curiosity_rng = np.random.default_rng(np.random.SeedSequence([seed, 908]))
        self.c=np.array(["conformity" in s["traits"] for s in specs])
        self.uniform_conformity = self.c & np.array([s.get("personalization") == "uniform" for s in specs])
        self.conformity_rng = np.random.default_rng(np.random.SeedSequence([seed, 907]))
        self.a=np.array(["aggressiveness" in s["traits"] for s in specs])
        self.o=np.array(["openness" in s["traits"] for s in specs])
        self.better_peer = np.array([s.get("peer_rule") == "better" for s in specs])
        self.any_peer = np.array([s.get("peer_rule") == "any" for s in specs])
        self.random_half_peer = np.array([s.get("peer_rule") == "random_half" for s in specs])
        self.peer_gate_rng = np.random.default_rng(np.random.SeedSequence([seed, 909]))
        self.current_peer_target = np.array([s.get("peer_target") == "current" for s in specs])
        self.riw=np.array([s.get("riw",False) for s in specs])
        self.mutation=np.array([s.get("mutation",False) for s in specs])
        self.reinitialize=np.array([s.get("reinitialize",False) for s in specs])
        self.eta=np.array([s.get("eta",settings["eta"]) for s in specs])
        self.alpha=np.array([s.get("alpha_mut",settings["alpha_mut"]) for s in specs])
        self.theta=np.array([s.get("theta",settings["theta"]) for s in specs])
        self.lam=np.array([s.get("lam",settings["lam"]) for s in specs])
        self.scores=tracker.evaluate(self.positions)
        self.pbests=self.positions.copy();self.pbest_scores=self.scores.copy()
        indices=self.pbest_scores.argmin(axis=1)
        self.global_best=self.pbests[np.arange(self.k),indices].copy()
        self.global_score=self.pbest_scores.min(axis=1)
        self.initial_best=self.global_score.copy()
        self.stagnation=np.zeros((self.k,population),dtype=int)
        self.reset_events=np.zeros(self.k,dtype=int)

    def step(self, observe_behavior=None):
        s=self.settings
        r1,r2=self.rng.rand(self.n,self.d),self.rng.rand(self.n,self.d)
        if not self.settings.get("coordinatewise_hppso_draws", False):
            hr1,hr2=self.hp_rng.rand(self.n,1),self.hp_rng.rand(self.n,1)
            r1=np.where(self.hp[:,None,None],hr1,r1);r2=np.where(self.hp[:,None,None],hr2,r2)
        inertia=np.where(self.a[:,None],.4+.5*self.traits[:,:,2],s["w"])
        inertia=np.where(self.riw[:,None],self.streams[1].uniform(.4,.9),inertia)
        cognitive=s["c1"]*r1*(self.pbests-self.positions)
        social=s["c2"]*r2*(self.global_best[:,None,:]-self.positions)
        conformity = self.traits[:, :, 1]
        if self.uniform_conformity.any():
            conformity = np.where(self.uniform_conformity[:, None], self.conformity_rng.random(), conformity)
        cognitive *= np.where(self.c[:, None], 1 - conformity, 1)[:, :, None]
        social *= np.where(self.c[:, None], conformity, 1)[:, :, None]
        inertia_term=inertia[:,:,None]*self.velocities
        self.velocities=inertia_term+cognitive+social
        peers=self.streams[2].integers(0,self.n-1,self.n);peers+=peers>=np.arange(self.n)
        opened=self.traits[:,:,3]>self.theta[:,None]
        probability=self.streams[3].random(self.n)<1-self.theta[:,None]
        if s["shared_openness"]=="probability":opened=np.where(self.shared[:,None],probability,opened)
        peer_scores = self.scores[:, peers]
        eligible = peer_scores > self.scores
        if self.better_peer.any():
            eligible = np.where(self.better_peer[:, None], peer_scores < self.scores, eligible)
        eligible |= self.any_peer[:, None]
        if self.random_half_peer.any():
            random_gate = self.peer_gate_rng.random(self.n) < 0.5
            eligible = np.where(self.random_half_peer[:, None], random_gate, eligible)
        enabled = self.o[:, None] & opened & eligible
        target = self.pbests[:, peers, :]
        if self.current_peer_target.any():
            target = np.where(self.current_peer_target[:, None, None], self.positions[:, peers, :], target)
        peer_term = np.where(enabled[:, :, None], self.lam[:, None, None] * (target - self.positions), 0)
        self.velocities += peer_term
        self.velocities=np.clip(self.velocities,-self.limit,self.limit)
        self.positions+=self.velocities
        uniforms=self.streams[4].random((self.n,self.d));noise=self.streams[5].standard_normal((self.n,self.d))
        curiosity = self.traits[:, :, 0]
        if self.uniform_curiosity.any():
            curiosity = np.where(self.uniform_curiosity[:, None], self.curiosity_rng.random(), curiosity)
        mask=self.q[:,None,None]&(uniforms<self.alpha[:,None,None]*curiosity[:, :, None])
        mutation_mask=uniforms[:,0]<s["pso_m_rate"]
        mask|=self.mutation[:,None,None]&mutation_mask[None,:,None]
        strength=np.where(self.mutation,s["pso_m_strength"],self.eta)
        if observe_behavior is not None:
            observe_behavior(inertia_term, cognitive, social, peer_term,
                             mask & (strength[:, None, None] != 0))
        self.positions+=np.where(mask,noise*strength[:,None,None]*self.width,0)
        reflect_bounds(self.positions.reshape(-1,self.d),self.velocities.reshape(-1,self.d),self.lower,self.upper)
        values=self.tracker.evaluate(self.positions)
        if self.tracker.count==self.tracker.budget:return
        self.scores=values
        improved=self.scores<self.pbest_scores
        self.pbest_scores[improved]=self.scores[improved];self.pbests[improved]=self.positions[improved]
        indices=self.pbest_scores.argmin(axis=1);rows=np.arange(self.k)
        better=self.pbest_scores[rows,indices]<self.global_score
        self.global_best[better]=self.pbests[rows[better],indices[better]]
        self.global_score=np.minimum(self.global_score,self.pbest_scores.min(axis=1))
        self.stagnation=np.where(improved,0,self.stagnation+1)
        reset=self.reinitialize[:,None]&(self.stagnation>=s["stagnation_threshold"])
        replacements=self.streams[6].random((self.n,4))
        self.traits=np.where(reset[:,:,None],replacements,self.traits)
        self.stagnation[reset]=0;self.reset_events+=reset.sum(axis=1)

    def optimize(self):
        while self.tracker.count<self.tracker.budget:self.step()
