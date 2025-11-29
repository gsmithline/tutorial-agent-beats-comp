import warnings
from typing import Tuple

import numpy as np

# Tolerance for regret feasibility
EPSILON: float = 1e-6


def _simplex_projection(x: np.ndarray) -> np.ndarray:
	"""
	Project onto probability simplex.
	"""
	x = np.asarray(x, dtype=float).reshape(-1)
	if (x >= 0).all() and abs(np.sum(x) - 1) < 1e-10:
		return x

	n = len(x)
	u = np.sort(x)[::-1]
	cssv = np.cumsum(u) - 1
	rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
	theta = cssv[rho] / (rho + 1)
	return np.maximum(x - theta, 0.0)


def compute_regret(mix: np.ndarray, game_matrix: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
	"""
	Regret for symmetric 2p game under mixture `mix` (row payoffs in game_matrix).
	- regrets[i] = max(0, (M[i]·mix) - (mix^T M mix))
	Returns (regrets, nash_value, expected_utils_per_pure).
	"""
	M = np.asarray(game_matrix, dtype=float)
	x = np.asarray(mix, dtype=float).reshape(-1)
	u_vals = M @ x
	nash_value = float(x @ u_vals)  # x^T M x
	regrets = np.maximum(0.0, u_vals - nash_value)
	return regrets, nash_value, u_vals


def milp_max_sym_ent_2p(game_matrix, discrete_factors: int = 100) -> np.ndarray:
	"""
	Compute maximum-entropy symmetric Nash equilibrium for a 2-player symmetric game.
	Uses CVXPY MIP with ECOS_BB, fallback to GLPK_MI. Based on the provided formulation.
	"""
	# Lazy import so the module can be imported without cvxpy installed
	try:
		import cvxpy as cp  # type: ignore
	except Exception as e:
		raise RuntimeError(f"cvxpy is required for MILP MENE solver: {e}")

	game_matrix_np = np.array(game_matrix, dtype=np.float64)
	if game_matrix_np.ndim != 2 or game_matrix_np.shape[0] != game_matrix_np.shape[1]:
		raise ValueError("game_matrix must be a square 2D array")

	# Fill NaNs column-wise with column mean, fallback to 0 if all NaN
	if np.isnan(game_matrix_np).any():
		for j in range(game_matrix_np.shape[1]):
			col = game_matrix_np[:, j]
			if np.isnan(col).any():
				col_mean = np.nanmean(col)
				if np.isnan(col_mean):
					col_mean = 0.0
				col_filled = np.where(np.isnan(col), col_mean, col)
				game_matrix_np[:, j] = col_filled

	M = game_matrix_np.shape[0]
	U = float(np.max(game_matrix_np) - np.min(game_matrix_np))
	if U <= 0:
		# Degenerate: all entries equal; any mixture works
		return _simplex_projection(np.ones(M) / M)

	x = cp.Variable(M)
	u = cp.Variable(1)
	z = cp.Variable(M)
	b = cp.Variable(M, boolean=True)

	obj = cp.Minimize(cp.sum(z))

	a_mat = np.ones((1, M))
	u_m = game_matrix_np @ x

	constraints = [
		u_m <= u + EPSILON,
		a_mat @ x == 1,
		x >= 0,
		u - u_m <= U * b,
		x <= 1 - b,
	]

	for k in range(discrete_factors):
		if k == 0:
			constraints.append(np.log(1 / discrete_factors) * x <= z)
		else:
			# linear approximation of x*log(x) at k/discrete_factors
			slope = ((k + 1) * np.log((k + 1) / discrete_factors) - k * np.log(max(k, 1) / discrete_factors))
			intercept = (k / discrete_factors) * np.log(max(k, 1) / discrete_factors)
			constraints.append(intercept + slope * (x - k / discrete_factors) <= z)

	prob = cp.Problem(obj, constraints)

	try:
		prob.solve(solver=cp.ECOS_BB)
		if not (prob.status and prob.status.startswith("optimal")):
			raise ValueError(f"ECOS_BB status: {prob.status}")
	except Exception as e:
		warnings.warn(f"Failed to solve with ECOS_BB: {e}")
		try:
			prob.solve(solver=cp.GLPK_MI)
			if not (prob.status and prob.status.startswith("optimal")):
				raise ValueError(f"GLPK_MI status: {prob.status}")
		except Exception as e2:
			raise RuntimeError(f"Both ECOS_BB and GLPK_MI solvers failed. ECOS_BB error: {e}, GLPK_MI error: {e2}")

	ne_strategy = _simplex_projection(np.array(x.value).reshape(-1))
	regret, _, _ = compute_regret(ne_strategy, game_matrix_np)
	max_regret = float(np.max(regret)) if regret.size else 0.0
	if max_regret <= EPSILON:
		return ne_strategy
	raise RuntimeError(f"Failed to find Nash equilibrium within {EPSILON} regret")


