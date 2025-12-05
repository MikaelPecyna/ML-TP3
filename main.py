from view.engine import Engine
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choisir le mode d'exécution")
    parser.add_argument(
        "-op", "--option",
        choices=["ql", "dl"],   # limite aux valeurs possibles
        required=True,
        help="Choisir le mode: ql pour Q-Learning, dl pour Deep Learning"
    )

    args = parser.parse_args()

    # Utilisation de l'argument
    engine = Engine(width=4, height=4)

    if args.option == "ql":
        engine.launcher = engine.launcherQL
    else:
        engine.launcher = engine.launcherDL

    engine.run()

