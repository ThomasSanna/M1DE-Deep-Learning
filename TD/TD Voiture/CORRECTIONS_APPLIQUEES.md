# Récapitulatif des corrections - Dataset Cars

## 🔍 Problèmes identifiés et corrections appliquées

### **Problème 1 : Colonnes one-hot encoded de type `bool` au lieu de `float`**

**Symptôme initial :**
- R² = -0.005 (le modèle ne fait que prédire la moyenne)
- MSE ≈ 1.0 sur données standardisées
- Prédictions constantes (toutes identiques)

**Cause :**
```python
# AVANT (pandas 2.0+ crée des bool par défaut)
data = pd.get_dummies(data, columns=...)
```
- Créait 382 colonnes `bool` + 6 colonnes `float64`
- `DataFrame.values` retournait un array de type **`object`**
- TensorFlow/Keras ne peut pas calculer les gradients sur un array `object`

**Correction (Cellule 23) :**
```python
# APRÈS
data = pd.get_dummies(data, columns=..., dtype=float)
```
✅ Toutes les colonnes sont maintenant `float64`

---

### **Problème 2 : Normalisation des colonnes one-hot encoded**

**Symptôme :**
- Loss ne descendait pas
- Prédictions constantes même après correction du dtype

**Cause :**
```python
# AVANT - Normalise TOUT !
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols.remove("Price")
train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
```
- Après `dtype=float` dans get_dummies, TOUTES les colonnes étaient float64
- StandardScaler normalisait les colonnes one-hot
- `Company Names_audi` passait de `[0, 1]` à `[-0.16, ...]` → détruit la signification binaire

**Correction (Cellules 24 + 29) :**

**Cellule 24** - Définir explicitement les colonnes numériques :
```python
# Liste les colonnes numériques D'ORIGINE (avant get_dummies)
numeric_features = ['HorsePower', 'Seats', 'Torque', 'Speed', 'Acceleration', 'Battery Capacity']
numeric_features = [col for col in numeric_features if col in data.columns]
```

**Cellule 29** - Scaler SEULEMENT ces colonnes :
```python
# APRÈS - Ne normalise QUE les vraies colonnes numériques
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
test_data[numeric_features] = scaler.transform(test_data[numeric_features])
```
✅ Les colonnes one-hot restent `[0.0, 1.0]`

---

### **Problème 3 : Valeurs manquantes (NaN) dans Battery Capacity**

**Symptôme :**
- Même après corrections 1 & 2, le modèle convergeait vers la moyenne
- Train Loss > Val Loss (anormal)

**Cause :**
- 2 NaN dans la colonne `Battery Capacity` (lignes 671 et 774 du train set)
- Keras/TensorFlow ne peut pas entraîner avec des NaN
- Propageait des NaN dans les gradients → modèle bloqué

**Diagnostic effectué :**
```python
X_train.isna().sum().sum()  # → 2
X_train.values.dtype  # → float64 (mais avec NaN, donc min/max/std = nan)
```

**Correction (Cellule 20) :**
```python
# Remplir les NaN dans Battery Capacity par la médiane
if 'Battery Capacity' in data.columns and data['Battery Capacity'].isna().sum() > 0:
    median_battery = data['Battery Capacity'].median()
    data['Battery Capacity'].fillna(median_battery, inplace=True)
```
✅ Plus aucun NaN dans le dataset

---

## 📊 Résultat des corrections

### Avant corrections :
- R² = **-0.005** (pire qu'une baseline)
- RMSE ≈ 0.95 (= prédire la moyenne)
- Prédictions : toutes identiques (variance ≈ 0)
- Loss stagne à ~1.0 (= variance de y_scaled)

### Après corrections :
- Le modèle devrait maintenant **apprendre correctement**
- Loss devrait descendre significativement < 0.5
- R² devrait être positif et élevé (> 0.6)
- Prédictions variables selon les inputs

---

## ✅ Liste des modifications dans l'ordre d'exécution

1. **Cellule 20** : Gestion des NaN dans Battery Capacity (remplacement par médiane)
2. **Cellule 23** : Ajout de `dtype=float` à `pd.get_dummies()`
3. **Cellule 24** : Définition explicite de `numeric_features` (6 colonnes seulement)
4. **Cellule 29** : Modification du scaling pour utiliser `numeric_features` au lieu de `numeric_cols`

---

## 🎯 Pour vérifier que tout fonctionne

Après ré-exécution depuis la cellule 20 :

```python
# 1. Vérifier les types
print(X_train.dtypes.value_counts())  # → Doit afficher "float64: 388"
print(X_train.values.dtype)  # → Doit afficher "float64" (pas "object")

# 2. Vérifier les NaN
print(X_train.isna().sum().sum())  # → Doit afficher 0

# 3. Vérifier les colonnes one-hot
onehot_col = [c for c in X_train.columns if '_' in c][0]
print(X_train[onehot_col].unique())  # → Doit afficher array([0., 1.])

# 4. Après entraînement, vérifier les métriques
print(f"R²: {r2:.4f}")  # → Doit être > 0.5
print(f"Train Loss finale: {history.history['loss'][-1]:.4f}")  # → Doit être < 0.3
```

---

## 📝 Leçons apprises

1. **Pandas 2.0+ change de comportement** : `get_dummies()` crée des `bool` par défaut → toujours spécifier `dtype=float` pour ML
2. **Ne JAMAIS normaliser les colonnes one-hot** : elles doivent rester binaires [0, 1]
3. **Toujours vérifier les NaN** AVANT le split train/test
4. **Diagnostiquer avec `dtype`** : un array numpy de type `object` indique un mélange de types
5. **Loss ≈ variance** = le modèle prédit la moyenne = il n'apprend rien

---

## 🗑️ Cellules de diagnostic à supprimer (optionnel)

Les cellules suivantes ont été ajoutées pour le diagnostic et peuvent être supprimées :
- Cellule 42 : Diagnostic X_train
- Cellule 43 : Test modèle avant entraînement
- Cellule 44 : Test entraînement 20 epochs
- Cellule 45 : Vérification NaN/Inf
- Cellule 46 : Localisation des NaN

Conserve la **cellule 20** (gestion des NaN) car elle fait partie de la correction permanente !
