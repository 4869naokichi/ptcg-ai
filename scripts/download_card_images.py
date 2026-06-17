from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.pokemontcg.io/v2/cards"
IMAGE_URL = "https://images.pokemontcg.io/{set_id}/{number}.png"
IMAGE_EXT = ".png"
SET_ID_BY_EXPANSION = {
    "SVE": "sve",
    "TEF": "sv5",
    "TWM": "sv6",
    "SCR": "sv7",
    "SSP": "sv8",
    "JTG": "sv9",
    "DRI": "sv10",
    "MEG": "me1",
}
ENERGY_NAMES = {
    "{G}": "Grass",
    "{R}": "Fire",
    "{W}": "Water",
    "{L}": "Lightning",
    "{P}": "Psychic",
    "{F}": "Fighting",
    "{D}": "Darkness",
    "{M}": "Metal",
}


@dataclass(frozen=True)
class LocalCard:
    card_id: int
    name: str
    expansion: str
    number: str
    category: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-only", action="store_true", help="Download images only for cards in submission/deck.csv.")
    parser.add_argument("--card-id", action="append", type=int, default=[], help="Download one local card ID. Can be repeated.")
    parser.add_argument("--all", action="store_true", help="Try every card in data/EN_Card_Data.csv.")
    parser.add_argument("--size", choices=["small", "large"], default="small")
    parser.add_argument(
        "--source",
        choices=["direct", "api", "auto"],
        default="direct",
        help="direct uses known images.pokemontcg.io set IDs; api searches the Pokémon TCG API; auto tries direct first.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing images.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve matches without downloading images.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Delay between API calls.")
    parser.add_argument("--api-timeout", type=float, default=8.0, help="Timeout for API search requests.")
    parser.add_argument("--image-timeout", type=float, default=30.0, help="Timeout for image download requests.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many selected cards.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "card_images")
    parser.add_argument("--card-data", type=Path, default=ROOT / "data" / "EN_Card_Data.csv")
    parser.add_argument("--deck", type=Path, default=ROOT / "submission" / "deck.csv")
    args = parser.parse_args()

    cards = read_card_data(args.card_data)
    selected_ids = select_card_ids(args, cards)
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = read_manifest(manifest_path)

    for card_id in selected_ids:
        card = cards.get(card_id)
        if card is None:
            print(f"skip {card_id}: missing from card data")
            continue

        output_path = args.output_dir / f"{card_id}{IMAGE_EXT}"
        if output_path.exists() and not args.force:
            print(f"skip {card_id}: already exists at {output_path}")
            continue

        direct_url = direct_image_url(card)
        if direct_url and args.source in {"direct", "auto"}:
            print(f"direct {card.card_id}: {card.name} ({card.expansion} {card.number}) -> {direct_url}")
            if not args.dry_run:
                try:
                    download_file(direct_url, output_path, timeout=args.image_timeout)
                except HTTPError as exc:
                    if args.source == "direct":
                        print(f"miss {card.card_id}: direct image returned HTTP {exc.code}")
                    else:
                        direct_url = None
                else:
                    manifest[str(card_id)] = manifest_record(card, image_url=direct_url, image_path=output_path.name)
                    write_manifest(manifest_path, manifest)
                    time.sleep(args.sleep)
                    continue
            else:
                time.sleep(args.sleep)
                continue

        if args.source == "direct":
            print(f"miss {card_id}: no direct set mapping for {card.expansion} {card.number}")
            time.sleep(args.sleep)
            continue

        try:
            match = resolve_card(card, timeout=args.api_timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"error {card_id} {card.name}: {exc}")
            time.sleep(args.sleep)
            continue

        if match is None:
            print(f"miss {card_id}: {card.name} ({card.expansion} {card.number})")
            time.sleep(args.sleep)
            continue

        image_url = match.get("images", {}).get(args.size) or match.get("images", {}).get("small")
        print(format_match(card, match, image_url))
        if not image_url:
            time.sleep(args.sleep)
            continue

        if not args.dry_run:
            download_file(image_url, output_path, timeout=args.image_timeout)
            manifest[str(card_id)] = manifest_record(
                card,
                match=match,
                image_url=image_url,
                image_path=output_path.name,
            )
            write_manifest(manifest_path, manifest)

        time.sleep(args.sleep)


def read_card_data(path: Path) -> dict[int, LocalCard]:
    cards: dict[int, LocalCard] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            card_id = int(row["Card ID"])
            if card_id in cards:
                continue
            cards[card_id] = LocalCard(
                card_id=card_id,
                name=row["Card Name"],
                expansion=row["Expansion"],
                number=row["Collection No."],
                category=row["Stage (Pokémon)/Type (Energy and Trainer)"],
            )
    return cards


def select_card_ids(args: argparse.Namespace, cards: dict[int, LocalCard]) -> list[int]:
    selected: list[int] = []
    if args.all:
        selected.extend(cards)
    if args.deck_only:
        selected.extend(read_deck_ids(args.deck))
    selected.extend(args.card_id)
    if not selected:
        selected.extend(read_deck_ids(args.deck))

    deduped = []
    seen = set()
    for card_id in selected:
        if card_id in seen:
            continue
        seen.add(card_id)
        deduped.append(card_id)
    return deduped


def read_deck_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def direct_image_url(card: LocalCard) -> str | None:
    set_id = SET_ID_BY_EXPANSION.get(card.expansion)
    if set_id is None:
        return None
    return IMAGE_URL.format(set_id=set_id, number=card.number)


def resolve_card(card: LocalCard, timeout: float) -> dict | None:
    for query in query_candidates(card):
        results = search_cards(query, timeout=timeout)
        if not results:
            continue
        scored = sorted(
            ((score_result(card, result), result) for result in results),
            key=lambda item: item[0],
            reverse=True,
        )
        if scored[0][0] > 0:
            return scored[0][1]
    return None


def query_candidates(card: LocalCard) -> list[str]:
    name = api_card_name(card.name)
    quoted = escape_query_value(name)
    candidates = [
        f'!name:"{quoted}" number:{card.number}',
        f'name:"{quoted}" number:{card.number}',
        f'!name:"{quoted}"',
        f'name:"{quoted}"',
    ]
    if "Energy" in name:
        candidates.append(f'supertype:Energy name:"{quoted}"')
    return candidates


def search_cards(query: str, timeout: float) -> list[dict]:
    params = {
        "q": query,
        "pageSize": "25",
        "orderBy": "-set.releaseDate,number",
        "select": "id,name,number,set,images,supertype,subtypes",
    }
    url = f"{API_URL}?{urlencode(params)}"
    request = Request(url, headers=request_headers())
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("data", [])


def download_file(url: str, output_path: Path, timeout: float) -> None:
    request = Request(url, headers=request_headers())
    with urlopen(request, timeout=timeout) as response:
        output_path.write_bytes(response.read())


def request_headers() -> dict[str, str]:
    headers = {"User-Agent": "ptcg-ai-local-visualizer/0.1"}
    api_key = os.environ.get("POKEMONTCG_IO_API_KEY")
    if api_key:
        headers["X-Api-Key"] = api_key
    return headers


def score_result(card: LocalCard, result: dict) -> int:
    score = 0
    result_name = result.get("name", "")
    if normalize(result_name) == normalize(api_card_name(card.name)):
        score += 100
    elif normalize(api_card_name(card.name)) in normalize(result_name):
        score += 50

    if normalize_number(result.get("number")) == normalize_number(card.number):
        score += 45

    card_set = result.get("set", {})
    ptcgo_code = card_set.get("ptcgoCode")
    if ptcgo_code and normalize(ptcgo_code) == normalize(card.expansion):
        score += 35
    if normalize(card.expansion) in normalize(card_set.get("id", "")):
        score += 12
    if normalize(card.expansion) in normalize(card_set.get("name", "")):
        score += 8

    if result.get("images", {}).get("small"):
        score += 10
    return score


def api_card_name(name: str) -> str:
    converted = name
    for token, energy_name in ENERGY_NAMES.items():
        converted = converted.replace(token, energy_name)
    converted = converted.replace("{", "").replace("}", "")
    converted = converted.replace("’", "'")
    return " ".join(converted.split())


def normalize(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def normalize_number(value: object) -> str:
    return str(value).lower().lstrip("0")


def escape_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def format_match(card: LocalCard, match: dict, image_url: str | None) -> str:
    card_set = match.get("set", {})
    set_label = card_set.get("ptcgoCode") or card_set.get("id") or card_set.get("name")
    image_label = "image" if image_url else "no-image"
    return (
        f"match {card.card_id}: {card.name} ({card.expansion} {card.number}) -> "
        f"{match.get('id')} {match.get('name')} ({set_label} {match.get('number')}) {image_label}"
    )


def manifest_record(
    card: LocalCard,
    image_url: str,
    image_path: str,
    match: dict | None = None,
) -> dict:
    return {
        "localCardId": card.card_id,
        "localName": card.name,
        "localExpansion": card.expansion,
        "localNumber": card.number,
        "apiId": match.get("id") if match else None,
        "apiName": match.get("name") if match else None,
        "apiSet": match.get("set", {}) if match else {},
        "apiNumber": match.get("number") if match else None,
        "imageUrl": image_url,
        "imagePath": image_path,
    }


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
