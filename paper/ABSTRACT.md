# Why twisted-hBN moiré artificial atoms cannot host orbital qubits: measured well deepening meets a two-phonon floor

Faruk Sami Bilgin (Independent Researcher, ORCID 0009-0001-9064-4410). Manuscript in preparation; extended abstract only.

**Question.** Building on the registry-resolved DFT measurement of the twisted-hBN moiré potential
(companion preprint, doi:10.5281/zenodo.24201814: cosine-only potential, P1 = −56.6, P2 = +17.6, P3 = +3.6 meV,
V0_eq = 40 meV at the PBE-D3 geometry; four-star curvature-constrained level spacing E21 = 32.4 meV at θ = 1.09°),
can the intra-well A1→E orbital transition serve as a qubit, i.e. can any accessible knob give it T1 ≥ 30 ns?

**Measured deepening (DFT on the same registry map).** Hydrostatic pressure deepens the well
(λ_V0 ≈ 0.70–0.80 Å; first-star weight ×1.15/1.32/1.53/1.75 at Δz = 0.1–0.4 Å, cosine-only form preserved,
Q1/P1 ≤ 2×10⁻⁴) and stiffens it (P2/P1 from −0.31 to −0.22), raising E21 from 32.4 to 48.2 meV at 3 GPa.
A third layer adds a factor ≈ √2 (×1.41). Biaxial strain (≤ 1% change per −2% strain) and perpendicular field (×0.99)
do not deepen. A direct six-point curvature scan at AA (+6.9 and +18.5 meV at 0.08a and 0.12a) shows AA is a true,
strongly curved minimum (harmonic spacing ≈ 40 meV). The AA–AB interlayer-gap corrugation falls from 0.263 Å
(ambient) to 0.222 Å (3 GPa) and saturates at 0.212–0.218 Å over 5–15 GPa, so a flattened well is unreachable.

**All-channel lifetime at T = 0.** Two independent implementations (plus an LA+LA cross-check on the companion
solver) include one-phonon LA (deformation, piezoelectric, pseudo-gauge, registry vertices), LA+LA with the full
intermediate-state resolvent, flexural ZA+ZA pairs, sequential two-phonon processes and flexural contact terms.
Two-phonon emission escapes the large-momentum form-factor suppression and dominates:
**T1 ≈ 6 ps at the measured well (band 1.4–50 ps), ≈ 11 ps at 3 GPa (3.4–60 ps), ≈ 8 ps at E21 = 64.4 meV (6–37 ps)**;
the floor is nearly independent of depth. The 30 ns target is missed by a factor ≈ 2600 at the working point
(≈ 520 in the most favourable corner), 600–5000 elsewhere. No lever — level spacing up to the optical-phonon bands,
the deformation potential (30 ns would need D ≲ 0.9 eV for LA+LA at 48 meV, vs 3.66–5.94 eV), phononic structuring
(dominant LA+LA wavelengths 2.7–5.4 nm), encapsulation, or all combined (T1 ~ 1 ns at best) — comes within a factor 30.

**Calibration and warning.** The same model class applied to the Si:P 2p0 state (measured T1 = 205 ± 18 ps)
gives 1.08 ns with real-tail envelopes (3.3–26× too long: omitted channels lengthen T1), and replacing the envelopes by
Gaussians of equal second moments lengthens it to 3.5×10⁴ s, a factor ~3×10¹³. A Gaussian form factor can manufacture
ten or more orders of magnitude of phonon protection; a one-phonon T1 of a deep bound state is an upper bound.

**Conclusion.** Orbital qubits in twisted hBN are ruled out. The identical giant-dipole infrared emitter interface,
spin/valley encodings, and a passive role for hBN (encapsulant; block-level sliding-ferroelectric offset, ±85 meV
by DFT) remain. An earlier internal draft that concluded otherwise rested on a one-phonon budget and two errors in
an LA+LA estimate (missing identical-boson factor 1/2; intermediate-state sum truncated at ten states).
