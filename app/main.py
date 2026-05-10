from rich.console import Console
from app.config import get_settings

console = Console()


def main():
    settings = get_settings()
    console.print("[bold green]AI Engineering Prep ready.[/bold green]")
    console.print(f"App: {settings.app_name}")
    console.print(f"Environment: {settings.app_env}")
    console.print(f"Log level: {settings.log_level}")


if __name__ == "__main__":
    main()
