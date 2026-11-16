"""Controlled trait experiments; production PSO/HPPSO implementations stay intact."""
import numpy as np
from ._pso_bounds import reflect_bounds, velocity_limit

TRAITS = ("curiosity", "conformity", "aggressiveness", "openness")


class TraitPSO:
    def __init__(self, objective, bounds, population, seed, spec, settings):
        self.objective, self.spec, self.settings = objective, spec, settings
        self.active = set(spec.get("traits", []))
        self.hp = spec["backbone"] == "hp"
        self.lower, self.upper = np.asarray(bounds, dtype=float)
        self.width = self.upper - self.lower
        self.n, self.d = population, len(self.lower)
        self.limit = velocity_limit(self.lower, self.upper)
        self.rng = np.random.RandomState(seed)
        self.streams = [np.random.default_rng(np.random.SeedSequence([seed, 900 + i])) for i in range(7)]
        self.conformity_rng = np.random.default_rng(np.random.SeedSequence([seed, 907]))
        self.positions = self.rng.uniform(self.lower, self.upper, (population, self.d))
        self.velocities = np.zeros_like(self.positions)
        self.traits = self.streams[0].random((population, 4))
        self.shared = {}
        for name in TRAITS:
            column = TRAITS.index(name)
            self.shared[name] = (float(self.traits[:, column].mean()) if settings["shared_level"] == "initial_mean"
                                 else float(settings["shared_level"]))
        self.scores = np.array([objective(x) for x in self.positions])
        self.pbests = self.positions.copy()
        self.pbest_scores = self.scores.copy()
        self.global_best = self.pbests[self.pbest_scores.argmin()].copy()
        self.global_score = float(self.pbest_scores.min())
        self.initial_best = self.global_score
        self.stagnation = np.zeros(population, dtype=int)
        self.reset_events = 0

    def values(self, name):
        if self.spec.get("personalization", "individual") == "shared":
            return np.full(self.n, self.shared[name])
        return self.traits[:, TRAITS.index(name)]

    def step(self):
        shape = (self.n, 1) if self.hp else (self.n, self.d)
        r1, r2 = self.rng.rand(*shape), self.rng.rand(*shape)
        s = self.settings
        inertia = s["w"]
        if "aggressiveness" in self.active:
            inertia = (0.4 + 0.5 * self.values("aggressiveness"))[:, None]
        if self.spec.get("riw", False):
            inertia = self.streams[1].uniform(.4, .9)
        cognitive = s["c1"] * r1 * (self.pbests - self.positions)
        social = s["c2"] * r2 * (self.global_best - self.positions)
        if "conformity" in self.active:
            conformity = (self.conformity_rng.random() if self.spec.get("personalization") == "uniform"
                          else self.values("conformity")[:, None])
            cognitive *= 1 - conformity
            social *= conformity
        self.velocities = inertia * self.velocities + cognitive + social
        if "openness" in self.active and self.n > 1:
            peers = self.streams[2].integers(0, self.n - 1, self.n)
            peers += peers >= np.arange(self.n)
            theta = self.spec.get("theta", s["theta"])
            if self.spec.get("personalization") == "shared" and s["shared_openness"] == "probability":
                opened = self.streams[3].random(self.n) < 1 - theta
            else:
                opened = self.values("openness") > theta
            active = opened & (self.scores[peers] > self.scores)
            self.velocities[active] += self.spec.get("lam", s["lam"]) * (self.pbests[peers[active]] - self.positions[active])
        self.velocities = np.clip(self.velocities, -self.limit, self.limit)
        self.positions += self.velocities
        if "curiosity" in self.active or self.spec.get("mutation", False):
            uniforms = self.streams[4].random((self.n, self.d))
            noise = self.streams[5].standard_normal((self.n, self.d))
            if self.spec.get("mutation", False):
                # Repository PSO-m: select particles, mutate all their coordinates.
                mask = np.broadcast_to((uniforms[:, 0] < s["pso_m_rate"])[:, None], self.positions.shape)
                strength = s["pso_m_strength"]
            else:
                mask = uniforms < self.spec.get("alpha_mut", s["alpha_mut"]) * self.values("curiosity")[:, None]
                strength = self.spec.get("eta", s["eta"])
            self.positions += np.where(mask, noise * strength * self.width, 0)
        reflect_bounds(self.positions, self.velocities, self.lower, self.upper)
        self.scores = np.array([self.objective(x) for x in self.positions])
        improved = self.scores < self.pbest_scores
        self.pbest_scores[improved] = self.scores[improved]
        self.pbests[improved] = self.positions[improved]
        i = self.pbest_scores.argmin()
        if self.pbest_scores[i] < self.global_score:
            self.global_score = float(self.pbest_scores[i])
            self.global_best = self.pbests[i].copy()
        if self.spec.get("reinitialize", False):
            self.stagnation = np.where(improved, 0, self.stagnation + 1)
            # Full matrix each generation: counterfactual arms share proposed resets.
            replacements = self.streams[6].random((self.n, 4))
            reset = self.stagnation >= s["stagnation_threshold"]
            self.traits[reset] = replacements[reset]
            self.stagnation[reset] = 0
            self.reset_events += int(reset.sum())

    def optimize(self):
        while True:
            self.step()
