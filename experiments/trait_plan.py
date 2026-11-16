"""Deduplicated variants and group membership for all four trait studies."""
from itertools import product
import hashlib
import json
from pathlib import Path
import numpy as np
from hppso.algorithms.trait_study import TRAITS
from experiments.trait_functions import SUPPORTED_FUNCTIONS

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["dimension"] < 2 or cfg["population"] < 2 or cfg["budget"] <= cfg["population"]:
        raise ValueError("Require dimension/population >=2 and budget > population")
    for key in ("dimension", "population", "budget", "trace_points", "bootstrap_resamples", "stagnation_threshold"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if len(cfg["seeds"]) != len(set(cfg["seeds"])) or not cfg["seeds"] or any(not isinstance(s,int) or not 0<=s<2**32 for s in cfg["seeds"]):
        raise ValueError("Seeds must be distinct unsigned integers")
    allowed = SUPPORTED_FUNCTIONS
    if not cfg["functions"] or len(set(cfg["functions"])) != len(cfg["functions"]) or not set(cfg["functions"]) <= set(allowed):
        raise ValueError("Choose distinct functions from the supported trait-study registry")
    if cfg["grid_mode"] not in ("full", "one_at_a_time") or cfg["shared_openness"] not in ("probability", "level"):
        raise ValueError("Invalid grid/shared-openness mode")
    if cfg["shared_level"] != "initial_mean" and (not isinstance(cfg["shared_level"], (int,float)) or not 0<=cfg["shared_level"]<=1):
        raise ValueError("Shared level must be in [0,1] or initial_mean")
    for key in ("alpha_mut", "theta", "pso_m_rate"):
        if not 0 <= cfg[key] <= 1: raise ValueError(f"Invalid {key}")
    for key in ("eta", "lam", "pso_m_strength", "w", "c1", "c2"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0: raise ValueError(f"Invalid {key}")
    for key in ("etas", "alphas", "thetas", "lambdas"):
        values=cfg[key]
        if not values or len(values)!=len(set(values)) or any(not np.isfinite(x) or x<0 for x in values):
            raise ValueError(f"Invalid grid: {key}")
        if key in ("alphas", "thetas") and any(x>1 for x in values):raise ValueError(f"Invalid grid: {key}")
    return cfg


def build_plan(cfg):
    variants, groups = {}, {}
    def add(group, label, backbone="pso", traits=(), **kw):
        spec=dict(backbone=backbone, traits=sorted(traits), **kw)
        if spec["traits"]:
            spec.setdefault("personalization", "individual")
        if "curiosity" in spec["traits"]:
            spec.setdefault("eta",cfg["eta"]);spec.setdefault("alpha_mut",cfg["alpha_mut"])
        if "openness" in spec["traits"]:
            spec.setdefault("theta",cfg["theta"]);spec.setdefault("lam",cfg["lam"])
        # Algebraically disabled openness is exactly the standard backbone.
        if spec["traits"] == ["openness"] and (kw.get("theta",cfg["theta"])>=1 or kw.get("lam",cfg["lam"])==0):
            spec=dict(backbone=backbone, traits=[])
        if not spec["traits"] and not spec.get("mutation") and not spec.get("riw"):
            spec=dict(backbone=backbone, traits=[])
        key="v_"+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
        variants.setdefault(key, dict(spec=spec, label=label))
        groups.setdefault(group, []).append(dict(variant=key, label=label, settings=dict(kw)))
        return key
    if cfg.get("groups") == ["hppso_openness_peer_rules"]:
        group = "hppso_openness_peer_rules"
        options = dict(backbone="hp", traits=TRAITS, reinitialize=True)
        baseline = add(group, "HPPSO", **options)
        add(group, "HPPSO-openness-better", **options, peer_rule="better")
        if cfg.get("random_peer_half_probability"):
            add(group, "HPPSO-openness-random (50%)", **options, peer_rule="random_half")
        else:
            add(group, "HPPSO-openness-random", **options, peer_rule="any")
        add(group, "HPPSO-worse-current", **options, peer_target="current")
        add(group, "HPPSO without openness", backbone="hp",
            traits=[trait for trait in TRAITS if trait != "openness"], reinitialize=True)
        return dict(variants=variants, groups=groups, factorial={}, no_reinitialization=None,
                    trait_order=list(TRAITS), baseline=baseline)
    if cfg.get("groups") == ["openness_peer_rules"]:
        group = "openness_peer_rules"
        baseline = add(group, "PSO")
        add(group, "PSO-openness-individual", traits=["openness"], personalization="individual")
        add(group, "PSO-open-better", traits=["openness"], peer_rule="better")
        add(group, "PSO-open-any", traits=["openness"], peer_rule="any")
        add(group, "PSO-openness-current", traits=["openness"], peer_target="current")
        return dict(variants=variants, groups=groups, factorial={}, no_reinitialization=None,
                    trait_order=list(TRAITS), baseline=baseline)
    if cfg.get("groups") == ["personality_refresh"]:
        group = "personality_refresh"
        baseline = add(group, "HPPSO", backbone="hp", traits=TRAITS, reinitialize=True)
        no_refresh = add(group, "HPPSO without personality reinitialization", backbone="hp",
                         traits=TRAITS, reinitialize=False)
        return dict(variants=variants, groups=groups, factorial={}, no_reinitialization=no_refresh,
                    trait_order=list(TRAITS), baseline=baseline)
    if cfg.get("full_hppso_sensitivity"):
        group, = cfg["groups"]
        options = dict(backbone="hp", traits=TRAITS, reinitialize=True)
        baseline = add(group, "HPPSO", **options)
        if group == "curiosity_grid":
            pairs = {(eta, cfg["alpha_mut"]) for eta in cfg["etas"]}
            pairs |= {(cfg["eta"], alpha) for alpha in cfg["alphas"]}
            for eta, alpha in sorted(pairs):
                add(group, f"HPPSO: eta={eta:g}, alpha={alpha:g}", **options, eta=eta, alpha_mut=alpha)
                groups[group][-1]["settings"]["personalization"] = "individual"
        elif group == "openness_grid":
            pairs = {(cfg["theta"], lam) for lam in cfg["lambdas"]}
            pairs |= {(theta, cfg["lam"]) for theta in cfg["thetas"]}
            for theta, lam in sorted(pairs):
                add(group, f"HPPSO: theta={theta:g}, lambda={lam:g}", **options, theta=theta, lam=lam)
                groups[group][-1]["settings"]["personalization"] = "individual"
        else:
            raise ValueError("Full HPPSO sensitivity requires a curiosity or openness grid")
        return dict(variants=variants, groups=groups, factorial={}, no_reinitialization=None,
                    trait_order=list(TRAITS), baseline=baseline)
    pso=add("curiosity", "PSO")
    add("curiosity", "PSO-m", mutation=True)
    for mode in ("individual", "shared"):
        personalization = "uniform" if mode == "shared" and cfg.get("curiosity_shared_per_iteration") else mode
        add("curiosity", f"PSO-curiosity-{mode}", traits=["curiosity"], personalization=personalization)
    for group in ("aggressiveness", "openness", "conformity"):
        add(group,"PSO")
    add("aggressiveness","PSO-aggressiveness",traits=["aggressiveness"])
    add("aggressiveness","PSO-RIW",riw=True)
    for mode in ("individual", "shared"):
        add("openness",f"PSO-openness-{mode}",traits=["openness"],personalization=mode)
        if mode == "individual" or cfg.get("conformity_shared_control", True):
            add("conformity",f"PSO-conformity-{mode}",traits=["conformity"],personalization=mode)
    if cfg.get("conformity_uniform_control", False):
        add("conformity", "PSO-conformity-uniform", traits=["conformity"], personalization="uniform")
    q_pairs = list(product(cfg["etas"],cfg["alphas"])) if cfg["grid_mode"]=="full" else sorted(set(
        [(e,cfg["alpha_mut"]) for e in cfg["etas"]]+[(cfg["eta"],a) for a in cfg["alphas"]]))
    o_pairs = list(product(cfg["thetas"],cfg["lambdas"])) if cfg["grid_mode"]=="full" else sorted(set(
        [(t,cfg["lam"]) for t in cfg["thetas"]]+[(cfg["theta"],l) for l in cfg["lambdas"]]))
    groups["curiosity_grid"] = [dict(e) for e in groups["curiosity"][:2]]
    groups["openness_grid"] = [dict(e) for e in groups["openness"][:1]]
    for mode in ("individual","shared"):
        for eta,alpha in q_pairs:
            personalization = "uniform" if mode == "shared" and cfg.get("curiosity_shared_per_iteration") else mode
            add("curiosity_grid",f"Q-{mode}: eta={eta:g}, alpha={alpha:g}",traits=["curiosity"],
                personalization=personalization,eta=eta,alpha_mut=alpha)
            groups["curiosity_grid"][-1]["settings"]["personalization"] = mode
        for theta,lam in o_pairs:
            add("openness_grid",f"O-{mode}: theta={theta:g}, lambda={lam:g}",traits=["openness"],
                personalization=mode,theta=theta,lam=lam)
    factorial={}
    for mask in range(16):
        traits=[name for i,name in enumerate(TRAITS) if mask&(1<<i)]
        label="HPPSO" if mask==15 else "HPPSO all traits off (factorial control)" if mask==0 else "HPPSO["+",".join(traits)+"]"
        if mask == 0 and cfg.get("factorial_pso_control"):
            factorial[str(mask)] = add("factorial", "PSO")
        else:
            factorial[str(mask)]=add("factorial",label,backbone="hp",traits=traits,reinitialize=True)
    groups["hp_ablation"]=[]
    for mask,label in [(15,"HPPSO")]+[(15^(1<<i),"HPPSO without "+name) for i,name in enumerate(TRAITS)]:
        groups["hp_ablation"].append(dict(variant=factorial[str(mask)],label=label,settings={}))
    no_reset = None
    if cfg.get("factorial_pso_control"):
        for group in ("curiosity", "aggressiveness"):
            groups["hp_ablation"].extend(e for e in groups[group] if e["label"] in ("PSO", "PSO-m", "PSO-RIW")
                                         and e["variant"] not in {a["variant"] for a in groups["hp_ablation"]})
    else:
        no_reset=add("hp_ablation","HPPSO without personality reinitialization",backbone="hp",traits=TRAITS,reinitialize=False)
    selected=cfg.get("groups")
    if selected:
        if set(selected)-set(groups):raise ValueError("Unknown experiment group")
        if "factorial" in selected or "hp_ablation" in selected:
            selected=list(dict.fromkeys(selected+["factorial","hp_ablation"]))
        groups={g:groups[g] for g in selected}
        used={e["variant"] for entries in groups.values() for e in entries}
        variants={k:v for k,v in variants.items() if k in used}
    return dict(variants=variants,groups=groups,factorial=factorial,
                no_reinitialization=no_reset,trait_order=list(TRAITS),baseline=pso)
