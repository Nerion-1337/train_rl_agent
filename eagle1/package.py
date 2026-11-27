"""Prépare le zip de dépôt des livrables, nommés selon la consigne : Nom_Prenom_n°_livrable_mmaaaa.

Lancement :  python -m eagle1.package --nom Dupont --prenom Marie --date 112026
Résultat   :  dist/Eagle1_Pilote_Automatique_Dupont_Marie.zip
"""
import argparse
import io
import zipfile

from eagle1.core import ROOT

DELIVERABLES = [  # (n°, nom du livrable, fichier source)
    (1, "Notebook_mission", "notebooks/mission_eagle1_lunarlander.ipynb"),
    (2, "Video", "media/eagle1_atterrissage.mp4"),
    (3, "API", "eagle1/api.py"),
    (4, "Interface_graphique", "eagle1/gui.py"),
    (5, "Tableau_de_bord", "eagle1/dashboard.py"),
    (6, "Rapport", "docs/rapport_eagle1.pdf"),
]
# Code source complet (les livrables .py importent eagle1.core) : de quoi tout relancer avec Docker
SOURCE = ["eagle1", "notebooks", "tests", "docs/build_report.py", "models/eagle1_best.zip", "models/eagle1_best.json",
          "Dockerfile", "docker-compose.yml", "pyproject.toml", "uv.lock", ".env", ".dockerignore", "README.md"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nom", required=True)
    parser.add_argument("--prenom", required=True)
    parser.add_argument("--date", required=True, help="date de démarrage du projet, format mmaaaa (ex. 112026)")
    parser.add_argument("--titre", default="Eagle1_Pilote_Automatique")
    args = parser.parse_args()

    prefix = f"{args.nom}_{args.prenom}"
    out = ROOT / "dist" / f"{args.titre}_{prefix}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for num, name, src in DELIVERABLES:
            path = ROOT / src
            zf.write(path, f"{prefix}_{num}_{name}_{args.date}{path.suffix}")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as code:
            for item in SOURCE:
                path = ROOT / item
                files = [path] if path.is_file() else [p for p in path.rglob("*")
                                                       if p.is_file() and "__pycache__" not in p.parts
                                                       and ".ipynb_checkpoints" not in p.parts]
                for f in files:
                    code.write(f, f.relative_to(ROOT).as_posix())
        zf.writestr(f"{prefix}_7_Code_source_{args.date}.zip", buf.getvalue())
    print(out)
    for info in zipfile.ZipFile(out).infolist():
        print(f"  {info.filename:60s} {info.file_size / 1024:8.0f} Ko")


if __name__ == "__main__":
    main()
