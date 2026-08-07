# Notation primer

Reference for the notation used across this project and the QEC literature.
Organizing principle: symbol → how to say it → what object it literally is →
what it is for here. Compressed mental model at the end.

## 1. States — Dirac notation

| Notation | Say it | What it literally is |
|---|---|---|
| \|ψ⟩ | "ket psi" | a column vector in C^N |
| ⟨ψ\| | "bra psi" | its conjugate transpose (row vector) |
| ⟨φ\|ψ⟩ | "bra-ket" | inner product → complex number |
| \|φ⟩⟨ψ\| | outer product | matrix (operator) |

Qubit = C^2. |0⟩ = (1,0)ᵀ, |1⟩ = (0,1)ᵀ. |ψ⟩ = α|0⟩ + β|1⟩ is the vector
(α, β); |α|²+|β|² = 1; amplitudes complex; global phase unphysical.
"Hilbert space" here = finite-dim complex vector space with inner product.

## 2. Many qubits — tensor product ⊗

C² ⊗ C² = C⁴; |01⟩ ≡ |0⟩⊗|1⟩. n qubits → 2ⁿ amplitudes (why statevector
simulation caps near 30 qubits on 8 GB GPUs). Entangled = not factorizable as
|a⟩⊗|b⟩; α|000⟩+β|111⟩ is the canonical example.

## 3. Operators and gates

U† ("dagger") = conjugate transpose. Unitary: U†U = I — all allowed closed-
system evolutions. Hermitian (A† = A) = observable; eigenvalues = outcomes.

Paulis: X = [[0,1],[1,0]] (bit flip), Z = [[1,0],[0,−1]] (phase flip — no
classical analogue; invisible to the repetition code), Y = iXZ. Each is
unitary AND Hermitian, squares to I, eigenvalues ±1 — same object serves as
error, gate, and observable. Z₁Z₂ = Z⊗Z⊗I (subscripts = positions);
weight = number of non-identity slots. H makes superpositions; CNOT entangles
(builds encoders and parity-check circuits).

## 4. Measurement — Born rule

Pr(0) = |⟨0|ψ⟩|² = |α|²; state collapses onto the outcome eigenspace.
Measuring Z₁Z₂ (eigenvalue +1 on |00⟩,|11⟩; −1 on |01⟩,|10⟩) IS the parity
question "same or different?" — it collapses only agree/disagree subspaces,
never individual bits (why checks are safe on codewords).
Convention: bits ↔ eigenvalues via 0 ↔ +1, 1 ↔ −1 (Stim uses bits).

## 5. Commutators

[A,B] = AB−BA (=0: compatible, simultaneously measurable).
{A,B} = AB+BA (=0: anticommute, AB = −BA). Two Paulis always do one or the
other: they anticommute iff they clash (X-type vs Z-type) on an odd number of
positions. THE key fact of error detection:
**error E flips check S iff E anticommutes with S.**

## 6. Stabilizer formalism

Code = commuting Pauli generators S₁..Sₘ; codespace = {|ψ⟩ : Sᵢ|ψ⟩ = +|ψ⟩ ∀i}.
Repetition code: {Z₁Z₂, Z₂Z₃}. Syndrome = vector of check outcomes.
Logical operators X̄, Z̄: commute with all generators, not products of them
(rep code: X̄ = X₁X₂X₃) — why a wrong decode (guess ⊕ truth = logical op) is
invisible afterward. Parameters [[n,k,d]] (double bracket = quantum):
n physical, k logical, d = min weight of a logical operator; corrects
⌊(d−1)/2⌋ errors. Rotated surface code = [[d², 1, d]].

## 7. Noise and rates

p = per-operation physical error rate; circuit-level depolarizing noise =
random non-identity Pauli w.p. p after each gate + measurement/reset flips.
p_L = logical error rate per shot (what every experiment here estimates).
p_th = threshold; below it p_L ~ A(p/p_th)^⌊(d+1)/2⌋ (exponential suppression
in d). Our MWPM surface-code crossing: p_th ≈ 0.007. T₁/T₂ = physical
relaxation/coherence times (upstream of p; central in nv-sensing-sim).

## 8. Classical/CS side

⊕ = XOR = addition mod 2; syndrome algebra is linear algebra over F₂ (=GF(2)):
s = He with parity-check matrix H. O(·) asymptotics: MWPM ≈ O(n³) worst case
(blossom on the matching graph: vertices = tripped detectors, edge weights =
−log chain probability). Stim vocabulary: detector = ⊕ of a check across
consecutive rounds (fires on change); observable = logical outcome bit;
detector error model (DEM) = F₂ map faults → detector/observable flips.
ML: logit ℓ, sigmoid σ(ℓ) = 1/(1+e^(−ℓ)), BCE loss, receptive field =
propagation span of the conv stack in grid cells.

## 9. Coming attractions

ρ = density matrix (pure: ρ = |ψ⟩⟨ψ|); Lindblad equations = continuous-time
noise on ρ (engine of nv-sensing-sim). ⟨A⟩ = ⟨ψ|A|ψ⟩ = expectation value.
e^{iHt} = Hamiltonian-generated evolution (rotation gates R_z(θ) = e^{−iθZ/2}).

## Compressed mental model

Kets are vectors, gates are unitaries, observables are Hermitians, Paulis are
all three at once, checks are commuting Paulis, errors are anticommuting ones,
and everything downstream of the syndrome is F₂ linear algebra + probability.
When a paper's notation disorients you, ask "which of these objects is this?"
