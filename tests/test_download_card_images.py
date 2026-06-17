from __future__ import annotations

import unittest

from scripts.download_card_images import LocalCard, api_card_name, direct_image_url, score_result


class DownloadCardImagesTest(unittest.TestCase):
    def test_api_card_name_converts_energy_tokens(self) -> None:
        self.assertEqual(api_card_name("Basic {W} Energy"), "Basic Water Energy")

    def test_score_prefers_matching_name_number_and_set_code(self) -> None:
        card = LocalCard(
            card_id=721,
            name="Kyogre",
            expansion="MEG",
            number="34",
            category="Basic Pokémon",
        )
        result = {
            "name": "Kyogre",
            "number": "34",
            "set": {"ptcgoCode": "MEG", "id": "meg"},
            "images": {"small": "https://example.test/kyogre.png"},
        }

        self.assertGreaterEqual(score_result(card, result), 190)

    def test_direct_image_url_uses_known_set_mapping(self) -> None:
        card = LocalCard(
            card_id=3,
            name="Basic {W} Energy",
            expansion="SVE",
            number="3",
            category="Basic Energy",
        )

        self.assertEqual(direct_image_url(card), "https://images.pokemontcg.io/sve/3.png")

    def test_direct_image_url_knows_mega_evolution_set(self) -> None:
        card = LocalCard(
            card_id=723,
            name="Mega Abomasnow ex",
            expansion="MEG",
            number="36",
            category="Stage 1 Pokémon",
        )

        self.assertEqual(direct_image_url(card), "https://images.pokemontcg.io/me1/36.png")


if __name__ == "__main__":
    unittest.main()
