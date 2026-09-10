"""Run with an installed wheel and python -I; never import from src/."""

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import trading_system
from trading_system.demo import create_demo_app


def main() -> None:
    package_path = Path(trading_system.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise AssertionError(f"Expected installed wheel, got {package_path}")
    with TemporaryDirectory(prefix="investment-demo-wheel-") as directory:
        app = create_demo_app(Path(directory))
        with TestClient(app) as client:
            page = client.get("/")
            assert page.status_code == 200 and "DEMO" in page.text
            assert client.get("/api/health").json()["provider_calls"] is False
            snapshot = client.get("/api/demo/snapshot").json()
            assert snapshot["effective"][0]["recommended_units"] == 250
            assert all(not row["actionable"] for row in snapshot["effective"])
            assert client.get("/resume").status_code == 200
            css = client.get("/assets/shell.css")
            assert css.status_code == 200 and "text/css" in css.headers["content-type"]
        with TestClient(create_demo_app(Path(directory))) as client:
            assert client.get("/api/demo/snapshot").json() == snapshot
    print("Installed wheel: demo HTML, CSS, constraints, and persisted snapshot OK")


if __name__ == "__main__":
    main()
