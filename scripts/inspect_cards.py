from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "src"))

from cg.api import CardType
from ptcg_ai.game.card_db import card_by_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("card_ids", nargs="+", type=int)
    args = parser.parse_args()

    for card_id in args.card_ids:
        card = card_by_id(card_id)
        if card is None:
            print(f"{card_id}: not found")
            continue
        print(f"{card.cardId}: {card.name} ({CardType(card.cardType).name})")
        for skill in card.skills:
            print(f"  skill {skill.name}: {skill.text}")
        for attack_id in card.attacks:
            print(f"  attack_id {attack_id}")


if __name__ == "__main__":
    main()
