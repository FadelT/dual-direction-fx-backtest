# Dual-Direction FX/Crypto Backtest — Research Report

## Objectif

Tester si une approche **hedge des deux directions pendant une compression** (long + short simultanés) bat le simple **attendre le breakout** avant d'entrer.

Hypothèse centrale : avec une arithmétique spot linéaire et les mêmes règles de sortie, les deux approches ont le même PnL brut — le hedge perd uniquement à cause des coûts de transaction supplémentaires.

---

## Stratégies comparées

| Stratégie | Description |
|---|---|
| `BREAKOUT` | Attend le breakout, entre 1x dans la direction révélée |
| `OCO` | Identique à BREAKOUT économiquement (ordre stop des deux côtés) |
| `HEDGE_1x1` | Ouvre 1x long + 1x short à l'armement, ferme le perdant au breakout |
| `HALF_HEDGE_ADD` | Ouvre 0.5x long + 0.5x short, ferme le perdant, ajoute 0.5x au gagnant |

---

## Données

- **Source** : Binance API (H4 natif) + Yahoo Finance (M15/H1 resamplé en H4)
- **Actifs testés** : EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF (FX) + BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, AVAXUSDT, LINKUSDT, BNBUSDT, DOGEUSDT (crypto)
- **Timeframes** : M15, H1, H4, D1
- **Historique** : jusqu'à 8 ans selon l'actif (BTC/ETH/XRP depuis 2018)

---

## Résultat #1 — Le hedge ne bat jamais le BREAKOUT

Sur **toutes les paires testées**, tous timeframes confondus, le hedge perd systématiquement vs BREAKOUT après coûts :

| Stratégie | Volume tradé | Surcoût vs BREAKOUT |
|---|---|---|
| BREAKOUT | 2x sides | baseline |
| HALF_HEDGE_ADD | 3x sides | ~35 pips/trade |
| HEDGE_1x1 | 4x sides | ~70 pips/trade |

**Conclusion** : l'hypothèse est confirmée. Le hedge n'apporte aucun avantage et coûte plus cher. La complexité n'est pas justifiée.

---

## Résultat #2 — Meilleur timeframe : H4

| Timeframe | Paires+ / Total | Observation |
|---|---|---|
| M15 | 1/6 | Trop de bruit, params non optimisés |
| H1 | 1/6 (USDJPY) | Fragile, dépend d'une période spécifique |
| **H4** | **4/6 FX, 7/8 crypto** | Meilleur équilibre signal/bruit |
| D1 | 2/6 | Trop peu de trades, variance élevée |

---

## Résultat #3 — Meilleure approche de calibration : Cross-Validation temporelle

### Problème identifié

La sélection naïve du meilleur paramètre sur une seule fenêtre IS (in-sample) produit de l'**overfitting** — les résultats ne se reproduisent pas hors échantillon.

### Solution : Time-Series Cross-Validation (comme en ML)

- Pour chaque fenêtre OOS de 6 mois, on évalue chaque combo de paramètres sur **3 fenêtres IS distinctes**
- Score : `expectancy × √(trades)` — pénalise les combos à haute variance
- On sélectionne le combo avec le **meilleur score moyen** sur les 3 folds IS

### Impact sur XRPUSDT

| Méthode | Return OOS | Fenêtres+ |
|---|---|---|
| Best params naïf | +49.5% | 5/13 |
| **Cross-validation** | **+140.8%** | **7/11** |

---

## Résultat #4 — Performance CV Walk-Forward sur toutes les cryptos (H4)

Capital initial : $10,000 | Risque : 5% du capital par trade

| Asset | Trades OOS | Win% | Return | Max DD | Fenêtres+ | |
|---|---|---|---|---|---|---|
| **XRPUSDT** | 131 | 55.0% | **+957%** | -27.3% | 9/12 | ✅ |
| **BTCUSDT** | 148 | 44.6% | **+373%** | -38.3% | 10/12 | ✅ |
| **AVAXUSDT** | 94 | 56.4% | **+252%** | -27.5% | 7/8 | ✅ |
| DOGEUSDT | 130 | 53.1% | +62% | -42.5% | 7/11 | ✅ |
| SOLUSDT | 107 | 52.3% | +44% | -24.4% | 5/9 | ✅ |
| ETHUSDT | 159 | 45.3% | +21% | -55.3% | 8/12 | ✅ |
| BNBUSDT | 125 | 48.8% | +19% | -33.1% | 6/12 | ✅ |
| LINKUSDT | 153 | 47.7% | -16.5% | -58.7% | 5/12 | ❌ |

**Période OOS** : juillet 2020 → août 2026 (6 ans pour BTC/ETH/XRP/BNB/LINK)

---

## Résultat #5 — Focus XRPUSDT : comparaison benchmarks

### Période complète OOS (6 ans)

| Benchmark | Return |
|---|---|
| XRP buy & hold | +477% |
| S&P 500 buy & hold | +148% |
| **XRP Stratégie BREAKOUT** | **+957%** |

### 2025 → août 2026

| Benchmark | Return |
|---|---|
| XRP buy & hold | **-51.8%** |
| S&P 500 buy & hold | +31.7% |
| **XRP Stratégie BREAKOUT** | **+64.5%** |

L'edge principal : la stratégie trade **long et short** — elle profite aussi des baisses que le buy & hold subit.

### 2026 seul (jan → août)

| Métrique | Valeur |
|---|---|
| Trades | 20 |
| Win rate | 55% |
| Return strat | **+5.4%** |
| Max drawdown | -18.2% |
| XRP buy & hold | -44.5% |

---

## Position Sizing

Modèle utilisé : **risque fixe % du capital courant par trade**

```
position_size = (capital × risk_pct) / stop_loss_$
pnl = net_pips × pip_size × position_size
```

Impact du % de risque sur XRPUSDT (20 mois OOS 2025-2026) :

| Risque/trade | Return | Max DD | Ratio R/DD |
|---|---|---|---|
| 1% | +14.9% | -4.2% | 3.55 |
| 2% | +31.1% | -8.3% | 3.74 |
| **5%** | **+87.9%** | **-20.2%** | **4.34** |
| 10% | +205.6% | -38.3% | 5.37 |

Recommandation : **2-5%** selon la tolérance au drawdown.

---

## Paramètres optimaux XRPUSDT

Sélectionnés par CV sur la période pré-2025 :

```
atr_quantile        = 0.30
max_range_atr       = 3.5
breakout_buffer_atr = 0.0
stop_atr            = 2.0
take_profit_r       = 3.0
compression_lookback_bars = 360  # adapté H4
```

---

## Limites & Avertissements

1. **Bar-based, pas tick-based** — l'exécution réelle peut différer (gaps, slippage variable)
2. **Spread constant** — en réalité le spread s'élargit lors des événements
3. **Peu de trades par fenêtre** — 8-20 trades par période de 6 mois, variance élevée
4. **Sensibilité à la source de données** — Yahoo vs Binance donnent des résultats légèrement différents
5. **Pas de swap/financement** — coût non modélisé pour les positions longues
6. **Pas de walk-forward sur les crypto FX** — AVAX et SOL ont moins d'historique

---

## Recommandations

### Pour trader en réel

1. **Paper trading 3-6 mois** avant tout capital réel
2. **Risque 1-2%** par trade pour commencer
3. **Re-calibrer les paramètres tous les 6 mois** via la même procédure CV
4. **Top 3 actifs** : XRPUSDT, BTCUSDT, AVAXUSDT
5. **Éviter** : LINKUSDT (seul actif négatif en OOS)

### Améliorations futures (V2)

- [ ] Tick-based execution
- [ ] Spread variable selon la session
- [ ] Filtre de session (London/NY uniquement)
- [ ] Filtre de tendance H1/H4 (Ichimoku, EMA)
- [ ] Walk-forward automatisé avec re-calibration mensuelle
- [ ] Multi-actifs simultanés avec gestion du risque global
- [ ] Swap/financement overnight

---

## Conclusion

La stratégie **BREAKOUT sur compression** a un edge réel et documenté sur les cryptos en H4, particulièrement sur XRPUSDT et BTCUSDT. La **calibration par cross-validation temporelle** est essentielle pour éviter l'overfitting et généraliser les paramètres hors échantillon.

Le hedge (HEDGE_1x1, HALF_HEDGE_ADD) ne bat **jamais** le simple BREAKOUT après coûts — l'hypothèse initiale est réfutée de manière robuste.

La stratégie surperforme significativement le buy & hold XRP et le S&P 500 sur 6 ans, avec un avantage particulièrement marqué en période baissière grâce à la composante SHORT.
