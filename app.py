import sys


def main() -> None:
    print("\nStyle Transfer (CycleGAN) - Launcher")
    print("1) Web UI (PyWebview)")
    print("2) Console (CLI)")
    print("0) Exit\n")

    choice = None
    if len(sys.argv) >= 2:
        # allow: python app.py web / python app.py cli
        arg = sys.argv[1].strip().lower()
        if arg in ("web", "w", "1"):
            choice = "1"
        elif arg in ("cli", "c", "2"):
            choice = "2"
        elif arg in ("0", "exit", "quit"):
            choice = "0"

    if choice is None:
        choice = input("Choose mode (1/2/0): ").strip()

    if choice == "1":
        from webview_app import run_web
        run_web()
        return

    if choice == "2":
        from styler.cli import run_cli
        run_cli()
        return

    print("Bye!")


if __name__ == "__main__":
    main()
