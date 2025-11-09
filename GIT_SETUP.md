# Instructions pour configurer Git et pousser votre code

## 1. Installer Git (si pas déjà installé)

Téléchargez Git depuis : https://git-scm.com/download/win
Installez-le avec les options par défaut.

## 2. Configurer Git (première fois seulement)

**IMPORTANT** : Utilisez `--global` pour configurer Git globalement (peut être fait depuis n'importe quel dossier).

Ouvrez PowerShell ou Git Bash et exécutez (depuis n'importe quel dossier) :

```bash
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

**Note** : Si vous voyez "Fatal: not in a git directory", c'est normal si vous n'avez pas encore initialisé le dépôt. Utilisez `--global` pour éviter cette erreur.

## 3. Initialiser le dépôt Git

**Naviguez d'abord vers votre dossier de projet** :

```bash
cd C:\Users\as_cu\Desktop\use_case
```

Puis initialisez le dépôt Git :

```bash
git init
git add .
git commit -m "Initial commit: Churn risk analysis and A/B test analysis"
```

## 4. Créer un dépôt privé sur GitHub

1. Allez sur https://github.com
2. Cliquez sur le "+" en haut à droite → "New repository"
3. Nommez votre dépôt (ex: "use_case" ou "churn-analysis")
4. **IMPORTANT** : Cochez "Private" pour garder le dépôt privé
5. **NE PAS** cocher "Initialize this repository with a README" (on a déjà un README)
6. Cliquez sur "Create repository"

## 5. Connecter votre dépôt local à GitHub

GitHub vous donnera des commandes. Exécutez-les (remplacez `USERNAME` et `REPO_NAME` par vos valeurs) :

```bash
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## 6. Partager le dépôt (optionnel)

Une fois le dépôt créé :
- Le dépôt est privé par défaut
- Pour partager : Settings → Collaborators → Add people
- Vous pouvez aussi générer un lien d'invitation

## Notes importantes

- Le fichier `.gitignore` exclut déjà :
  - Les fichiers Python compilés (`__pycache__`)
  - L'environnement virtuel (`venv_use_case/`)
  - Les fichiers de sortie PDF dans `output/`
  - Les fichiers de configuration IDE

- Si vous voulez inclure les fichiers CSV de données, commentez la ligne dans `.gitignore` :
  ```
  # ressource/*.csv
  ```

