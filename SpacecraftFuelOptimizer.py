import os
import math
import time
import numpy as np

# gymnasium + RL

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from typing import Optional
# astrodynamics
from astropy import units as u
from astropy.time import TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver

"""
OrbitalTransferEnv
------------------
Goal: Minimize total Δv to move from a circular LEO (e.g., 700 km) to another circular LEO (e.g., 800 km).
- Coplanar, two-body Earth model (no J2).
- Continuous action: small 3D impulsive Δv per step in RTN (local orbital) frame, bounded each step.
- Each step:
    1) Apply impulsive Δv (fuel -= |Δv|).
    2) Propagate for dt seconds.
- Episode ends on success (close to target circular orbit) or if out of fuel / steps.
- Reward: -|Δv| each step, with shaping on |a - a_target| and eccentricity to encourage circularization.
This is intentionally simple and fast to train as a portfolio demo.
"""

def norm(v):
    return float(np.linalg.norm(v))

class OrbitalTransferEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        init_alt_km_range=(650.0, 750.0),
        target_alt_km_range=(780.0, 820.0),
        step_dt_s=120.0,
        max_steps=600,
        max_step_dv=5.0,     # m/s per step (action bound)
        fuel_budget=250.0,   # total m/s allowed
        success_a_tol_m=1000.0,     # semi-major axis tolerance (meters)
        success_e_tol=1e-3,          # eccentricity tolerance (dimensionless)
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.step_dt_s = step_dt_s
        self.max_steps = max_steps
        self.max_step_dv = max_step_dv
        self.fuel_budget = fuel_budget
        self.success_a_tol_m = success_a_tol_m
        self.success_e_tol = success_e_tol

        # Action: 3D Δv in RTN (m/s) bounded per step
        self.action_space = spaces.Box(
            low=-self.max_step_dv,
            high= self.max_step_dv,
            shape=(3,),
            dtype=np.float32
        )

        # Observation:
        # [a_err (km), e, fuel (m/s), step/ max_steps, norm_pos_err(km), norm_vel_err(km/s)]
        # plus raw elements: [a (km), ex, ey] to give the policy some direct signals
        # (we keep it compact; this is a didactic environment)
        high = np.array([np.inf, 1.0, np.inf, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.mu = Earth.k.to_value(u.km**3 / u.s**2)  # Earth's GM in km^3/s^2 
        self.earth_radius_km = Earth.R.to_value(u.km)

        self._episode_reset()

    # helpers:
    def _sample_alt_km(self, r):
        lo, hi = r
        return float(self.rng.uniform(lo, hi))

    def _random_raan_argperi_nu(self):
        # Keep coplanar i = 0 for simplicity. Randomize angles for diversity.
        raan = float(self.rng.uniform(0, 2*np.pi))
        argp = float(self.rng.uniform(0, 2*np.pi))
        nu   = float(self.rng.uniform(0, 2*np.pi))
        return raan, argp, nu

    def _make_circular(self, alt_km):
        a = (self.earth_radius_km + alt_km) * u.km
        return Orbit.circular(Earth, alt=a)

    def _rtn_frame(self, orbit):
        # Build RTN basis from current (r,v)
        r_vec, v_vec = orbit.rv()
        r = r_vec.to_value(u.km)
        v = v_vec.to_value(u.km/u.s)
        r_hat = r / norm(r)
        h = np.cross(r, v)
        n_hat = h / norm(h)
        t_hat = np.cross(n_hat, r_hat)
        return r_hat, t_hat, n_hat, r, v

    def _apply_impulse_rtn(self, orbit, dv_rtn_mps):
        r_hat, t_hat, n_hat, r, v = self._rtn_frame(orbit)
        dv_vec_kmps = (dv_rtn_mps / 1000.0) * (r_hat + 0*t_hat)  # placeholder
        # Compose dv in km/s using RTN basis
        dv_vec_kmps = (dv_rtn_mps[0]/1000.0)*r_hat + (dv_rtn_mps[1]/1000.0)*t_hat + (dv_rtn_mps[2]/1000.0)*n_hat
        man = Maneuver.impulse(dv_vec_kmps * (u.km / u.s))
        new_orbit = orbit.apply_maneuver(man)
        return new_orbit

    def _propagate(self, orbit, dt_s):
        return orbit.propagate(TimeDelta(dt_s, format="sec"))

    def _elements_summary(self, orbit):
        a_km = orbit.a.to_value(u.km)
        e = float(orbit.ecc.value)
        return a_km, e

    def _obs(self):
        a_km, e = self._elements_summary(self.orbit)
        a_t_km, e_t = self._elements_summary(self.target)
        a_err_km = (a_km - a_t_km)

        # Position/velocity error at this epoch 
        r, v = self.orbit.rv()
        rt, vt = self.target.rv()
        pos_err_km = norm((r - rt).to_value(u.km))
        vel_err_kmps = norm((v - vt).to_value(u.km/u.s))

        obs = np.array([
            a_err_km,
            e,
            self.fuel_remaining,
            self.steps / self.max_steps,
            pos_err_km,
            vel_err_kmps,
            a_km,
            math.cos(2*np.pi*e),  # crude feature to help circularization
            math.sin(2*np.pi*e)
        ], dtype=np.float32)
        return obs

    def _success(self):
        a_km, e = self._elements_summary(self.orbit)
        a_t_km, _ = self._elements_summary(self.target)
        return (abs(a_km - a_t_km)*1000.0 <= self.success_a_tol_m) and (e <= self.success_e_tol)

    def _episode_reset(self):
        init_alt = self._sample_alt_km(self.init_alt_km_range if hasattr(self, "init_alt_km_range") else (650.0, 750.0))
        tgt_alt  = self._sample_alt_km(self.target_alt_km_range if hasattr(self, "target_alt_km_range") else (780.0, 820.0))
        self.init_alt_km_range = getattr(self, "init_alt_km_range", (650.0, 750.0))
        self.target_alt_km_range = getattr(self, "target_alt_km_range", (780.0, 820.0))

        self.orbit  = self._make_circular(init_alt)
        self.target = self._make_circular(tgt_alt)

        # Randomize true anomaly for both to avoid trivialization
        raan, argp, nu = self._random_raan_argperi_nu()
        T = self.orbit.period.to_value(u.s)
        tshift = float(self.rng.uniform(0, T))
        self.orbit = self._propagate(self.orbit, tshift)

        Tt = self.target.period.to_value(u.s)
        tshift_t = float(self.rng.uniform(0, Tt))
        self.target = self._propagate(self.target, tshift_t)

        self.fuel_remaining = self.fuel_budget
        self.steps = 0

    # gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._episode_reset()
        return self._obs(), {}

    def step(self, action):
        self.steps += 1

        # Clip and apply Δv
        dv = np.clip(np.array(action, dtype=np.float32), -self.max_step_dv, self.max_step_dv)
        dv_mag = norm(dv)

        if dv_mag > 0.0 and self.fuel_remaining > 0.0:
            if dv_mag > self.fuel_remaining:
                dv = dv * (self.fuel_remaining / (dv_mag + 1e-8))
                dv_mag = norm(dv)
            self.orbit = self._apply_impulse_rtn(self.orbit, dv)
            self.fuel_remaining -= dv_mag

        # Propagate dynamics
        self.orbit = self._propagate(self.orbit, self.step_dt_s)

        # Compute reward
        a_km, e = self._elements_summary(self.orbit)
        a_t_km, _ = self._elements_summary(self.target)
        a_err_km = abs(a_km - a_t_km)

        reward = -dv_mag  # pay for fuel
        # shaping: encourage approaching target a and circularizing
        reward += -0.0005 * a_err_km          # scale small to avoid overshadowing dv cost
        reward += -2.0 * max(0.0, e - self.success_e_tol)  # penalize eccentricity above tolerance

        # Check termination
        terminated = False
        truncated = False

        if self._success():
            reward += 100.0  # big terminal bonus
            terminated = True

        if self.fuel_remaining <= 0.0:
            truncated = True  # out of fuel

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._obs()
        info = {
            "dv_used": self.fuel_budget - self.fuel_remaining,
            "a_km": a_km,
            "e": e,
            "a_t_km": a_t_km,
            "a_err_km": a_err_km,
            "success": self._success()
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        a_km, e = self._elements_summary(self.orbit)
        a_t_km, _ = self._elements_summary(self.target)
        print(f"Step {self.steps} | a={a_km:.1f} km (target {a_t_km:.1f}) e={e:.4f} fuel={self.fuel_remaining:.1f} m/s")


def train_and_evaluate(
    total_timesteps=200_000,
    n_envs=8,
    exp_dir="runs/orbital_rl",
    model_name="ppo_orbital",
    seed=42
):
    os.makedirs(exp_dir, exist_ok=True)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(exp_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Vectorized env for throughput
    def make_env():
        env = OrbitalTransferEnv(
            init_alt_km_range=(680, 740),
            target_alt_km_range=(790, 810),
            step_dt_s=120.0,
            max_steps=500,
            max_step_dv=5.0,
            fuel_budget=220.0,
            success_a_tol_m=1000.0,
            success_e_tol=1e-3,
            seed=seed
        )
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)

    # PPO with small MLP
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        n_steps=2048 // n_envs,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=log_dir,
    )

    print("Training...")
    start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    dur = time.time() - start
    print(f"Training complete in {dur/60:.1f} min")

    save_path = os.path.join(models_dir, f"{model_name}.zip")
    model.save(save_path)
    print(f"Saved model to {save_path}")

    # Evaluation
    eval_env = OrbitalTransferEnv(
        init_alt_km_range=(690, 710),
        target_alt_km_range=(795, 805),
        step_dt_s=120.0,
        max_steps=500,
        max_step_dv=5.0,
        fuel_budget=220.0,
        success_a_tol_m=1000.0,
        success_e_tol=1e-3,
        seed=seed + 999
    )

    episodes = 50
    successes = 0
    dv_list = []
    overhead_list = []

    # Compute theoretical Hohmann Δv for circular coplanar transfer 
    def hohmann_delta_v(a1_km, a2_km):
        mu = eval_env.mu  # km^3/s^2
        r1 = a1_km
        r2 = a2_km
        v1 = math.sqrt(mu / r1)
        v2 = math.sqrt(mu / r2)
        a_t = 0.5 * (r1 + r2)
        v_peri_trans = math.sqrt(mu * (2/r1 - 1/a_t))
        v_apo_trans  = math.sqrt(mu * (2/r2 - 1/a_t))
        dv1 = abs(v_peri_trans - v1)
        dv2 = abs(v2 - v_apo_trans)
        return (dv1 + dv2) * 1000.0  # m/s

    print("Evaluating...")
    for ep in range(episodes):
        obs, _ = eval_env.reset()
        info_start = {}
        a0_km, _ = eval_env._elements_summary(eval_env.orbit)
        at_km, _ = eval_env._elements_summary(eval_env.target)
        dv_theory = hohmann_delta_v(a0_km, at_km)

        total_dv = 0.0
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = eval_env.step(action)
            total_dv = info["dv_used"]
            if term or trunc:
                if info["success"]:
                    successes += 1
                break

        dv_list.append(total_dv)
        if dv_theory > 1e-6:
            overhead_list.append(100.0 * max(0.0, (total_dv - dv_theory)) / dv_theory)

    success_rate = 100.0 * successes / episodes
    dv_mean = float(np.mean(dv_list))
    dv_std  = float(np.std(dv_list))
    overhead_mean = float(np.mean(overhead_list)) if overhead_list else float("nan")
    overhead_std  = float(np.std(overhead_list))  if overhead_list else float("nan")

    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {episodes}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Δv used (mean ± std): {dv_mean:.1f} ± {dv_std:.1f} m/s")
    print(f"Δv overhead vs. Hohmann (mean ± std): {overhead_mean:.2f}% ± {overhead_std:.2f}%")

    
    print("\nSuggested resume line (update numbers with your run):")
    print(f"Fuel Optimization RL: Achieved ~{success_rate:.0f}% success with Δv overhead ~{overhead_mean:.1f}% vs. Hohmann across {episodes} eval scenarios.")

if __name__ == "__main__":
  
    train_and_evaluate(
        total_timesteps=200_000,
        n_envs=8,
        exp_dir="runs/orbital_rl",
        model_name="ppo_orbital",
        seed=42
    )
