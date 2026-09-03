"""Entry point: ``python -m ai_world`` or the ``ai-world`` script."""
from __future__ import annotations


def main() -> None:
    from ai_world.ui.app import App

    App().run()


if __name__ == "__main__":
    main()
