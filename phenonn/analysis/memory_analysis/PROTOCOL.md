# Protocole expérimental — Angle « Mémoire » de PhenoNN

Objectif d'ensemble : **montrer que PhenoNN, entraîné sur le LAI satellite, apprend une mémoire
météo→phénologie (effets différés / legacy) que les modèles process-based ratent**, la quantifier
(longueur / patron / force, cadre SAM [28 Ogle]), et reproduire deux résultats-clés :
la **legacy de sécheresse sur le SOS de l'année suivante** [22 Liu Y.] et l'**effet différé du
printemps chaud sur le LAI d'été** [02 Buermann].

## 0. Rappels techniques PhenoNN (contraintes de design)
- **Entrée** : fenêtre de **720 j** = concat(année Y−1, année Y), derniers 720 j (≈ 10 janv. Y−1 → 31 déc. Y).
  Features journalières ERA5 (Tmin/Tmax/Tmean, ssrd_sum, strd_sum, tp_sum, VPD_max/mean, Rn_tot, PET)
  + co2 + 15 fractions PFT. Normalisées (z-score, log1p sur queues lourdes).
- **Sortie** : **36 valeurs dékadaires de LAI de l'année Y** (prédites en une passe).
- Conséquence mémoire : une perturbation en **Y−1** est **antérieure** au SOS de Y (≈ dékad 10–12, début avril) →
  legacy inter-annuelle **causale** et testable.
- **⚠️ Causalité intra-fenêtre** : la sortie de Y est produite à partir de TOUTE la fenêtre, y compris
  l'été de Y (postérieur au SOS de Y). Donc :
  - Pour la **legacy depuis Y−1** (exp. M3) : la perturbation est entièrement avant le SOS(Y) → **propre**.
  - Pour l'**analyse de sensibilité** (M1) : n'interpréter causalement que les **lags antérieurs** à la
    dékad cible, et **privilégier les architectures causales** (`bitransformer_v2`/`attnlstm`, `causal=True`).
- **Limite structurelle** : PhenoNN ne voit que la **météo** → il capte la mémoire **exogène** (Ogle),
  **pas** l'endogène (réserves NSC, stress imprint [49, 50]). Les effets mesurés sont un **plancher**.
  Variante à tester : ajouter un **proxy d'humidité du sol** (`dataset_05/add_soil_moisture_proxy.py`)
  pour renforcer le canal mémoire.

## Boîte à outils commune (à coder une fois dans `study/memory_analysis/`)
1. **Extraction de métriques phéno** depuis la courbe LAI (36 dékads → interpolation quotidienne
   double-logistique via `interpolate_lai_daily.py`), méthode **seuil** [D Bórnez] :
   - amplitude A = max − min de l'année ; **SOS = 1er jour où LAI > min + 0.30·A** (branche montante),
     **EOS = dernier jour où LAI > min + 0.40·A** (branche descendante), **LOS = EOS − SOS**,
     **peak** (jour du max), **∫LAI** (intégrale saisonnière, proxy de productivité).
2. **Runner contrefactuel à l'inférence** : charger un checkpoint, éditer le **window météo brut**
   (avant normalisation) sur une plage de dates/variables, puis prédire. Mécanisme non-invasif calqué
   sur `study/feature_ablation/train_ablate.py` (hook au niveau du dataset, `phenon/` non touché).
3. **Opérateur sécheresse** `D(window; fenêtre_temporelle, sévérité s)` :
   `tp_sum *= (1−s)` sur les jours ciblés (+ option `VPD_max/mean *= 1+s'`, `PET *= 1+s'`).
4. **Stratificateurs** : PFT (via fraction dominante), **aridité** WI = Σtp_sum / ΣPET (seuil 0.65,
   dryland/humide [21]), latitude.
5. **Baselines de contraste** : (a) **XGBoost** météo→LAI sans mémoire longue (`phenon/xgb_train.py`,
   features ~concomitantes), (b) modèle **process/GDD** simple. Servent à montrer que la mémoire est
   propre à PhenoNN.

Données : utiliser les **années de validation** (hors entraînement) et `selected_pixels`, pour l'honnêteté.

---

## M1 — Noyau de mémoire (cadre SAM [28]) — *fondation*
**Hypothèse** : la sensibilité du LAI d'une dékad d à une perturbation météo au lag j décroît avec j,
avec une **longueur de mémoire** non nulle (> quelques semaines) et des **lags** identifiables.

**Méthode** :
- Cible = une dékad d (p.ex. dékad du SOS, du pic, de l'EOS).
- **Jacobien par autograd** : `∂LAI_d / ∂(feature v, jour t)` sur toute la fenêtre 720 j (une backward pass).
  Alternative robuste : différences finies (perturbation +1 z-score à un seul jour t).
- Balayer t → courbe **S_d,v(lag)** ; agréger sur N sites × années, **par PFT** et **par variable v**.

**Métriques** :
- **Longueur** L = lag au-delà duquel |S| < 5 % de son max.
- **Patron** : position des pics (instantané ~lag 0 vs lags saisonniers ~30/90/365 j).
- **Force** : Σ|S| sur les lags « mémoire » (> 30 j) / Σ|S| total = **fraction mémoire**.

**Critères de succès** : fraction mémoire nettement > 0 ; noyau qui décroît proprement ;
**mémoire plus longue en drylands** (carte de L, figure originale).

---

## M2 — Décomposition instantané vs mémoire (gain de skill)
**Hypothèse** : allonger la fenêtre / garder le passé améliore la prédiction (mémoire utile),
surtout en **interannuel** (cf. +18–28 % de variance chez Ogle).

**Méthode (2 variantes)** :
- **A. Troncature de fenêtre** (réentraînement) : entraîner PhenoNN à 720 / 365 / 180 / 90 j →
  **gain de R² vs longueur** = contribution mémoire. Coûteux mais net.
- **B. Masquage à l'inférence** (sans réentraîner) : remplacer la partie **antérieure** du window
  (> 90 j avant chaque dékad) par la **climatologie** du site → chute de skill = contribution mémoire.

**Métriques** : ΔR² **global / site-année / interannuel** entre mémoire complète et mémoire ablatée ;
ΔRMSE sur SOS/EOS.

**Critères de succès** : ΔR² mémoire > 0 (viser quelques points) ; **l'interannuel** est le plus dégradé
par l'ablation de mémoire (signature attendue).

---

## M3 — Legacy de sécheresse sur le SOS de l'année suivante ([22]) — *expérience phare*
**Hypothèse** : une sécheresse en Y−1 **retarde le SOS prédit de l'année Y** (obs. +1.2 à +2.3 j),
plus fortement en drylands, de façon monotone avec la sévérité — **là où les modèles process ne le font pas**.

**Méthode** :
1. **Baseline** : prédire l'année Y sur météo réelle → SOS_base(Y), sur N sites × années.
2. **Contrefactuel** : appliquer `D(·, JJA de Y−1, s)` (déficit ≥ 2 mois consécutifs, à la [22]),
   Y inchangée → SOS_drought(Y). **ΔSOS = SOS_drought − SOS_base**.
3. **Stratifier** par aridité (WI) et PFT ; **dose-réponse** en balayant s ∈ {0.2, 0.4, 0.6, 0.8}.
4. **3 types de sécheresse [22]** en jouant sur la persistance du déficit jusqu'au **printemps de Y**
   (type 1 = déficit non résorbé au printemps Y → retard max ; type 3 = résorbé tôt → faible).
5. **Contraste modèle** : même contrefactuel sur XGBoost / GDD → doivent montrer **~0 retard**.

**Métriques** : distribution de ΔSOS (moyenne, % pixels retardés), par WI/PFT ; pente ΔSOS vs sévérité ;
ΔSOS(type1) vs ΔSOS(type3).

**Critères de succès** : **ΔSOS > 0** (ordre ~1–3 j) ; **plus fort en drylands** ; **monotone** en sévérité ;
**type1 > type3** ; **baseline process ≈ 0** → *PhenoNN capture la legacy exogène manquante*.

**Pièges** : effet borné au canal exogène (météo héritée) — magnitude = plancher ; tester la variante
**+ proxy SM** pour renforcer. Vérifier que le déficit Y−1 est bien dans la fenêtre (il l'est).

---

## M4 — Effet différé du printemps chaud sur le LAI d'été ([02])
**Hypothèse** : un printemps chaud produit un **effet différé négatif** sur le LAI d'été via stress
hydrique, sur ~10–15 % des surfaces (obs.), **contrairement aux DGVM** (qui prédisent surtout du positif).

**Méthode** :
- **A. Corrélation partielle** (réplique directe [02]) sur le **LAI prédit** : sur la période, corrélation
  partielle T_printemps ↔ LAI_été (en contrôlant le climat estival concomitant) par pixel →
  **carte du signe** de la legacy (schéma +/−/0). Comparer aux ~15 % négatif / ~5 % positif.
- **B. Contrefactuel** : `+ΔT` (ex. +2 °C) sur **MAM de Y uniquement** → trajectoire LAI été (JJA) de Y ;
  mesurer l'**anomalie tardive**.

**Métriques** : % pixels à legacy négative ; dépendance du signe aux **précipitations/altitude**
(leur random forest) ; anomalie LAI JJA.

**Critères de succès** : présence de régions à **legacy négative** (~10–15 %), concentrées en zones
**limitées par l'eau/altitude** ; signe **piloté par la précipitation**.

---

## M5 (optionnel) — Carte globale de la longueur de mémoire
À partir de M1 : cartographier **L** par pixel/PFT → drylands = mémoire longue. **Figure de couverture**.

---

## Ordre suggéré & dépendances
1. **Boîte à outils** (extraction phéno + runner contrefactuel + opérateur sécheresse).
2. **M1** (noyau) → valide que la mémoire existe et la caractérise (peu coûteux, autograd).
3. **M3** (legacy sécheresse) → le résultat phare, s'appuie sur la boîte à outils.
4. **M2** (gain de skill) et **M4** (printemps) → renforcent et généralisent.
5. **M5** (carte) → figure finale.

## Livrables attendus
- 1 module `study/memory_analysis/` (extraction phéno, runner contrefactuel, opérateur sécheresse).
- Figures : noyau S_d(lag) par PFT ; carte de longueur de mémoire ; ΔSOS vs sévérité/aridité ;
  carte du signe de la legacy printanière ; table ΔR² instantané/mémoire.
- Comparaison systématique à ≥ 1 baseline sans mémoire longue (XGBoost/GDD).

## Cadrage / limites à écrire dans l'article
- Mémoire **exogène** seulement ; pas d'endogène (NSC) → magnitudes = plancher ([49, 50, 28]).
- Cible **LAI** (structure de canopée), pas GCC ni flux carbone → « réponse phénologique ».
- Architectures : privilégier `causal=True` pour l'interprétation de sensibilité.
- Dépendance d'échelle des métriques ([29]) + mismatch grain 0.1° (météo) / 0.05° (cible).
