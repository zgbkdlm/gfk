import jax
import jax.numpy as jnp


def kf(state_trans, state_cov,
       obs_trans, obs_cov,
       m0, v0, ys):
    def scan_body(carry, elem):
        mp, vp = carry
        y = elem

        # Update
        S = obs_trans @ vp @ obs_trans.T + obs_cov
        chol = jax.scipy.linalg.cho_factor(S)
        K = jax.scipy.linalg.cho_solve(chol, obs_trans @ vp).T

        mf = mp + K @ (y - obs_trans @ mp)
        vf = vp - K @ S @ K.T

        # Prediction
        mp = state_trans @ mf
        vp = state_trans @ vf @ state_trans.T + state_cov
        return (mp, vp), (mf, vf)

    return jax.lax.scan(scan_body, (m0, v0), ys)[1]
