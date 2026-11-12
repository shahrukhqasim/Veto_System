# Datasets: muons and neutrinos from the target, transport, stores, forced interactions

Status 2026-11-12 (Shah Rukh + Claude). The datasets behind the veto design numbers (docs/STATUS.md): how they were made,
how they are weighted, where they are. All paths are on the farm (`/disk/lhcb_data` = `/disk/gfs_lhcb`).

## 0. Normalisation (read this first)

- **1 spill = 4e13 protons on target** (FairShip `run_fixedTarget.py` default). **15 years = 1.5e7 spills = 6e20 p.o.t.**
- Every `weight` column is **particles (or events) per spill**. 15-year numbers = per-spill numbers x 1.5e7.
- What sums to one spill:

| Object | Weights sum to | So when you use several |
|---|---|---|
| one muon pool chunk (`train_NNN.h5`, `hidden_NNN.h5`) | 1 spill (4.69e10 muons) | N chunks summed = N spills: divide by N |
| one transported seed (`shell/train/seed_S.npz`) | 1 spill | N seeds summed = N spills: divide by N (each seed is a "run") |
| a dataset store (`data/store/design-optimization-dataset-*`) | 1 spill in total | already divided by the number of seeds (`run` = seed index) |
| a neutrino store (`store2026_nu<flavour>`) | 1 spill | - |
| an event store (`evstore/<name>`) | events per spill | - |

- A chunk is a **thinned spill**: its ~9.7e7 rows carry the rates of a full spill (mean weight ~480 real muons per row), not
  4.69e10 real muons. The physical statistics of the whole muon pool are those of the simulated protons (table below), i.e.
  ~7.5e-5 of a spill of minimum bias. Rates, occupancies and pair rates are right on average; rare tails need forcing (section 5).

## 1. Muons at the end of the target (Nov-4 production)

Made on CSCS Eiger, 4 Nov 2026 (~102 node-hours), `tailtransport` in "complete" mode (`cpp/tailgen`, `bin/tail_produce.py`).

| Setting | Value |
|---|---|
| Event generator | Pythia 8.223 (the 2018 version; also the decayer of prompt / heavy-flavour hadrons, `Check:event` on) |
| Transport in the target | **Geant4 11.2, FTFP_BERT**, muons recorded where they leave the target (z = -0.0045 m) |
| Target | FairShip default `target_config.yaml` (19 W plates, Ta cladding, He gaps, Inconel vessel, steel enclosure, Cu proximity shielding) |
| Beam | FairShip: Gaussian sigma 0.8 cm + painting radius 5 cm |
| Threshold | muons with E > 10 GeV kept; p >= 400 GeV dropped as unphysical (0 rows in this pool) |
| Charm / beauty | Pythia 6 cascade (`bin/charm_cascade.py`, fracp 0.40 for W, chi x 1.200 A-scaling), decays by Pythia 8 (`bin/hf_decay.py`) |
| Validation | vs analog Geant4 (1e8 p.o.t.): 1.003; FairShip 2018 sample reproduced to 0.99-1.03 per bin when its 2018 weighting is applied (docs/STATUS.md, 1 Nov) |

**Components.** Each is normalised to its own simulated protons: weight per spill = w_decay / rep x 4e13 / protons
(`bin/build_pool_v5.py`); "protons" is the `events` entry in `pool.json`.

| Component | Slices x protons | Protons simulated | Muons / spill | Share |
|---|---|---|---|---|
| min. bias (+ gamma / dimuon, all non-heavy-flavour sources) | 6000 x 5e5 | **3.0e9** | 3.95e10 | 84 % |
| charm (forced: every proton makes charm, the weight carries the probability) | 3000 x 1e4 | 3.0e7 | 7.2e9 | 15 % |
| quarkonia (J/psi, psi', Upsilons) | 2500 x 2e7 | 5.0e10 | 1.9e8 | 0.4 % |
| beauty (forced) | 400 x 2.5e4 | 1.0e7 | 2.4e6 | ~0 |
| **total** | | | **4.69e10** | |

**Importance sampling in the production** (all unbiased, checked against unweighted runs):
- Russian roulette / weight window on the cascade secondaries (min. bias): flux 0.995 +- 0.009.
- Decay splitting: each long-lived or prompt decay that can give a muon is repeated (`rep`) with weight 1/rep: split 10
  (min. bias, charm, beauty), 20 (quarkonia).
- Forced decays: pi, K, K_L, K_S, Lambda, Sigma, Xi, ... decay with their decay probability as weight (Geant4's own decay
  kinematics where Geant4 has the channel; K_S / Lambda / Sigma- / Xi semileptonic channels added); prompt eta, omega, rho,
  phi, eta', J/psi, D, B, tau decayed by Pythia; gamma -> mu mu and e+ e- -> mu mu boosted.
- Muon weight window per (p, pT) cell (F = 0.1): ~0.6 rows per collision instead of ~20, n_eff 99.9 %.

**The pool** (`/disk/lhcb_data/sqasim/nov4_pool/`, 145 GB):
- `train_000.h5 .. train_022.h5` (23 chunks) and `hidden_000.h5 .. hidden_003.h5` (4 chunks, 15 % of the source events, by a
  hash of (component, slice, event); never used for training). **2,620,986,919 rows**, ~9.7e7 per chunk (5 GB each).
- Each chunk is scaled per component to the whole pool's flux, so **each chunk is one spill on its own** (section 0).
- Columns: px py pz x y z, pdg (PDG id: 13 = mu-), weight (per spill), pool_index (row id in the whole pool), component
  (0 min. bias, 1 onia, 2 charm, 3 beauty), gen (process code), parent (parent PDG), p_prod, ch (decay channel).
- `pool.json`: protons per component, flux per spill, rows and scale per chunk. `farm.md5` / `eiger.md5`: checksums.
- Raw production (354 GB, generator files, logs, windows, commands): only on CSCS
  `/capstor/scratch/cscs/sqasim/target-production-complete-november-4-2026` (**scratch purge ~4 Dec 2026**); 15 GB copied to
  `/disk/lhcb_data/sqasim/nov4_raw`.

## 2. Muons through the shield and the cavern (cuda-muons transport)

`bin/generate_muon_study.py --storage shell` (cuda-muons via Guglielmo's MuonsAndMatter, read only from
`/disk/users/gfrise/MuonsAndMatter`), container `/disk/users/lprate/containers/MuonsAndMatterContainer.sif`, farm-gpu3/4/5.

- Settings: shield parameters TRY6, diluted iron, nuclear interactions from B, cavern, simulated field, 5000 steps, every
  muon to every plane. States stored at the start (target exit), z = 30.6 m and z = 82 m; bounce point (kx, ky, kz), walls hit.
- **Seed S uses chunk S mod 23** (train) / S mod 4 (hidden), each with its own random stream: **3,220 train + 560 hidden
  seeds** (~140 per chunk) in `/disk/lhcb_data/sqasim/nov4_pool/shell/{train,hidden}/seed_S.npz` (106 GB). **Each seed is one
  spill.** The heavy rows are different muons in different seeds (MCS, energy loss), so seeds add statistics
  near-linearly up to ~10-20 per chunk.
- Muon types (`muon_types.TYPES`): 0 sst1_bounced, 1 sst1_selected (p > 1 GeV in the timing acceptance), 2 sst1_other,
  3 veto_only, 4 rock_bounced, 5 straight_past, 6 stops_between, 7 stops_before.
- Shell storage (`lib/veto/shell.py`: an elliptic arch 1.5 m around the vessel, z 29.89 -> 94.5 m, 32 x 16 side cells + 8 x 8
  caps): types 0-3 and muons stopping in the last metre of the shield kept with p = 1 (cap 2e5 per seed); muons crossing the
  shell kept with p = min(1, lam w / V_cell) so every cell gets about the same number of equally weighted rows (~1e5 per seed);
  plain Horvitz-Thompson weights w / p (`weight_raw`, `p_keep` also stored). ~2.4e5 rows / 28 MB per seed. Check:
  `bin/shell_check.py` (bias 1.002).
- Per spill: 4.69e10 from the target, 4.38e10 stop in the shield, 3.11e9 reach 30.6 m (1.46e9 into the -x wall, 9.1e7 into the
  floor), ~3e4 reach SST1.

**The design store** used by the evaluations:
`/disk/lhcb_data/sqasim/workspace/uzh/Veto_System/veto/data/store/design-optimization-dataset-nov4-shell-300seeds-2026-11-10`
(`bin/shell_to_store.py`): the first 300 train seeds, 7.17e7 rows, weights / 300 (sums to 4.72e10 muons per spill),
`run` = seed (combinatorial pairs are formed within a run).

**Charge convention** (fixed 2026-11-11): the pool's `pdg` is a PDG id (13 = mu-); the transported seeds' and stores' `pdg` is
the **charge sign (+1 = mu+)**. Always read it with `veto.muon_types.charge_from_pdg_column` (handles both). Regression check:
`bin/check_charge_convention.py`. Older stores (alpha .. zeta, c, sst1/ubt_TRY6) have the same convention; results from before
the fix used the wrong sign.

## 3. Neutrinos from the target (2026 production)

`tailtransport` in neutrino mode (same FairShip target, **Geant4 11.2 FTFP_BERT**, Pythia 8.223, hadrons followed above
1.1 GeV), `bin/nu_prod.sbatch`, 9 Nov 2026. Every neutrino with E >= 1 GeV within 0.3 rad, recorded where it is made.
Details and validation: `docs/neutrinos_2026.md`.

| Component | Slices x protons | Protons simulated | Note |
|---|---|---|---|
| min. bias (`nuMB_nu_*`) | 500 x 2e4 | 1e7 | rows with heavy-flavour ancestors dropped (counted by the next two) |
| charm (`charmc_nu_*`, forced) | 1000 x 1e4 | 1e7 | charm hadrons > 1 GeV, 50 % kept (weight x 2) |
| beauty (`beautyc_nu_*`, forced) | 400 x 2.5e4 | 1e7 | from the Nov-4 beauty hadrons, 5 % kept |

Weight per spill = w_decay x 4e13 / (protons x rep): each row stands for ~4e6 real neutrinos.
Raw: `/disk/lhcb_data/sqasim/nu_prod/tail/`. Validated against Guglielmo's GSimple 2026 flux (nu_mu 1-4 %) and SHiP 2025
(arXiv 2504.06692).

**Stores** (`/disk/lhcb_data/sqasim/workspace/uzh/Veto_System/veto/data/interactions/neutrinos/store2026_nu<flavour>`;
states at 30.6 m and 82 m, |x|, |y| <= 10 m):

| Flavour | Rows | Per spill |
|---|---|---|
| nu_mu | 24,091,408 | 1.33e11 |
| anti-nu_mu | 22,052,607 | 1.06e11 |
| nu_e | 2,946,292 | 1.52e10 |
| anti-nu_e | 3,352,653 | 1.51e10 |
| nu_tau | 196,212 | 1.28e9 |
| anti-nu_tau | 197,691 | 1.28e9 |
| total | 52,836,863 | |

## 4. Interaction libraries

| Library | Generator | File |
|---|---|---|
| muon DIS (helium, air, PVC, plane materials) | Geant4 below nu_x(E), Pythia 6 above ("crossover" split) | `data/dis/mu_library.npz` (`fastveto.dis.MuonLibrary`) |
| muon DIS in the shield iron (MS7) | **Geant4 11.2.2**, forced muonNuclear (Guglielmo's) | `mu_library_iron.npz` (split geant4) |
| neutrino interactions | **GENIE 3.06.02**, tune G18_02a_00_000, 15 nuclei (H .. Fe56), 6 flavours, 36 energies 1-500 GeV, 2000 events each | `data/interactions/genie/nu_library_v2.npz` |
| n / anti-n / K_L inelastic on helium | **Geant4 11.2 FTFP_BERT** (`cpp/g4hadlib`), forced, 24 energies 1.5-300 GeV x 20k | `/disk/lhcb_data/sqasim/neutral_lib/neutral_he.npz` |
| muon DIS in the cavern rock | Geant4 (`cpp/g4rock`, G4_CONCRETE half space, forced muonNuclear) + Pythia 6 part | `/disk/lhcb_data/sqasim/rock_dis/library/` |
| EM debris (e, gamma) out of the shield | Geant4 (`bin/em_debris.py`) on the Nov-4 parents | `results/em_debris/em_maps.npz` |

## 5. Forced interactions and event stores

Interactions are rare (~1e-6 per muon in the helium), so they are **forced**: each muon (or neutrino) crossing a target
volume is drawn with probability ~ w x P (P = n sigma L, the interaction probability), the vertex uniform along the crossing,
and each draw carries the weight S / M (S = sum of w P = interactions per spill, M = draws). Secondary particles that matter
(K_L, K_S, Lambda) are then forced to decay in the vessel, with weight x P(decay), and copied several times ("V0 copies",
"re-decays"). Low-statistics corners get extra draws (`--region corner`) or synthetic samples aimed at the dangerous window
(`bin/coverage_bound.py klmap --window`).

Design-independent **event stores** (`lib/veto/evstore.py`, `/disk/lhcb_data/sqasim/evstore/`): an event table (weight per
spill, code, parent row, vertex, time, forced-copy link, V0 pdg) + a track tree (parent, pdg, start, momentum, end, end reason,
creator), propagated straight with decays in flight to the cavern wall / z = 95 m; optional Geant4 downstream (`g4vessel`,
`bin/g4_vessel_transport.py`). Event codes (`interactions.CODE_NAMES`): 0 helium, 1 air, 2 pvc, 3 shield_iron, 4
shield_iron_v0, 5 rock_wall, 6 rock_floor, 7 neutral_he; 10 + i = plane i of the design.

| Store | What | Forcing |
|---|---|---|
| `ms7_muon_g4v112_nov4` (+ `_redecay100`) | muon DIS in the last 50 cm of MS7, parents from our transport of the Nov-4 pool (`/disk/lhcb_data/sqasim/ms7_parents/muons_iron_nov4.npz`, 4.62e8 muons / spill through the iron) | 10 x 2e6 interactions, 5 copies per V0; 100 re-decays of the 14,168 pre-selected V0 |
| `ms7_muon_g4v112` | the same with Guglielmo's 2018 parents | as above |
| `ms7_nu2026` | neutrino DIS in MS7 | 5e5 interactions, 100 V0 copies |
| `vessel_mu_nov4q_D6_straw_mod` | muon DIS in the helium, air, PVC, planes (after the charge fix) | 7e6 draws |
| `vessel_nu2026_D6_straw_mod_nu<flavour>` | neutrino DIS in the vessel and planes | per flavour |
| `rock_muon`, `rock_corner` | muon DIS in the cavern rock (300 spills, every exit; corner: floor z < 70 m, wall z < 52 m) | 2.2e6 / 1.7e6 draws |
| `rock_muon_nhe`, `rock_corner_nhe` | neutrons / K_L from the rock interacting in the helium | 50 copies per root |
| `rock_nu2026` | neutrino DIS in the rock | 4.1e5 events |

## 6. Software versions in one place

| Step | Software |
|---|---|
| p + target, hadron cascade, muons / neutrinos out of the target | Pythia 8.223 + Geant4 11.2 FTFP_BERT (tailtransport); charm / beauty: Pythia 6 cascade + Pythia 8 decays |
| muons through the shield and the cavern | cuda-muons (MuonsAndMatter, TRY6) on GPU |
| muon DIS in the shield (MS7) | Geant4 11.2.2 (cross-check Geant4 10.7.3, FairShip's: consistent) |
| neutrino interactions | GENIE 3.06.02, G18_02a_00_000 |
| neutral hadrons in the helium, rock, EM debris, Geant4 downstream | Geant4 11.2 |
| analysis | container `/disk/lhcb_data/sqasim/images/cudabear_november_2025.sif` (via `images/shared/exec_cudabear_november_2025.sh`) |

## 7. Older datasets

`data/store/design-optimization-dataset-{alpha,gamma,delta,epsilon,zeta,c}-*` (Oct 2026) come from older pools (v1-v5, based on
the 2018 h5 or thinner productions) with the old (2018-like) source spectrum; dataset c has 3-4x more SST1 muons above
30-100 GeV near the axis. Use the Nov-4 store for new numbers. All share the charge-sign convention of section 2.
