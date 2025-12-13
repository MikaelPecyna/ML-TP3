#!/bin/python3

from view.engine import Engine
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choisir le mode d'exécution")
    parser.add_argument(
        "-op", "--option",
        choices=["ql", "dl", "ddqn"],   # limite aux valeurs possibles
        required=True,
        help="Choisir le mode: ql pour Q-Learning, dl pour Deep Learning, ddqn pour Double Deep Q-Learning"
    )

    args = parser.parse_args()

    # Utilisation de l'argument
    engine = Engine(width=4, height=4)

    if args.option == "ql":
        engine.launcher = engine.launcherQL
    elif args.option == "dl":
        engine.launcher = engine.launcherDL
    else:  # ddqn
        engine.launcher = engine.launcherDDQN



    engine.run()

