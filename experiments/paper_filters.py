import numpy as np
from scipy import signal


def generate_paper_filter(family, num_points=512, order=16):
    """Generate paper family C or D with conjugate pole/zero pairs."""
    if family not in {"C", "D"}:
        raise ValueError("family must be C or D")
    if order < 2 or order % 2:
        raise ValueError("order must be a positive even integer")

    sections = order // 2

    def sample_roots():
        u = np.random.uniform(0.0, 1.0, sections)
        radius = np.sqrt(u) if family == "C" else u
        radius = np.minimum(radius, np.nextafter(1.0, 0.0))
        angle = np.random.uniform(0.0, np.pi, sections)
        return radius * np.exp(1j * angle)

    zeros = sample_roots()
    poles = sample_roots()

    sos = np.column_stack([
        np.ones(sections),
        -2 * zeros.real,
        np.abs(zeros) ** 2,
        np.ones(sections),
        -2 * poles.real,
        np.abs(poles) ** 2,
    ])

    _, h = signal.sosfreqz(sos, worN=num_points)

    return np.abs(h), np.unwrap(np.angle(h)), h.real, h.imag, sos