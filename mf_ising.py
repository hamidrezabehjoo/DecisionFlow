import numpy as np


class MFIsingSolver:
    """Naive mean-field fixed point for Ising: m = tanh(h + J m)."""

    def __init__(self, J, h=None, max_iter=10000, tol=1e-10, damping=0.5):
        self.N = J.shape[0]
        self.J = np.array(J, dtype=np.float64)
        self.h = np.zeros(self.N) if h is None else np.array(h, dtype=np.float64)
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping

        self.m = np.zeros(self.N, dtype=np.float64)
        self._run_mf()

    def _run_mf(self):
        m = np.clip(self.m, -0.999999, 0.999999)
        for _ in range(self.max_iter):
            field = self.h + self.J @ m
            m_new = np.tanh(field)
            m_next = (1.0 - self.damping) * m + self.damping * m_new
            if np.max(np.abs(m_next - m)) < self.tol:
                m = m_next
                break
            m = m_next
        self.m = np.clip(m, -0.999999, 0.999999)

    def marginal(self, i):
        """Return approximate marginal {+1: p, -1: 1-p} from MF magnetization."""
        p_plus = 0.5 * (1.0 + float(self.m[i]))
        p_plus = min(max(p_plus, 0.0), 1.0)
        return {+1: p_plus, -1: 1.0 - p_plus}

