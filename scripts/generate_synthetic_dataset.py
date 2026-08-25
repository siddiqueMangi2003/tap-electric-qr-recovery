import argparse
from pathlib import Path

from app.ml.synthetic_dataset import SyntheticDatasetConfig, SyntheticDatasetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate labelled synthetic EV-charger sticker scans."
    )
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--chargers", type=int, default=100)
    parser.add_argument("--variants-per-charger", type=int, default=8)
    parser.add_argument("--blur-variants-per-charger", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticDatasetConfig(
        chargers=args.chargers,
        variants_per_charger=args.variants_per_charger,
        blur_variants_per_charger=args.blur_variants_per_charger,
        seed=args.seed,
    )
    manifest_path = SyntheticDatasetGenerator(config).generate(args.output)
    count = config.chargers * config.variants_per_charger
    print(f"Generated {count} examples at {manifest_path}")


if __name__ == "__main__":
    main()
