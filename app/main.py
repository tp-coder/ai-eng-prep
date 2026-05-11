import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from app.config import get_settings
from app.llm import LLMClient, LLMConfigurationError, LLMResponseParsingError
from app.logging_config import configure_logging


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Engineering Prep CLI")
    parser.add_argument("prompt", nargs="?",
                        help="Prompt to send to the configured LLM provider")
    return parser.parse_args()


def render_items(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "none"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings)

    if not args.prompt:
        console.print("[bold green]AI Engineering Prep ready.[/bold green]")
        console.print(f"App: {settings.app_name}")
        console.print(f"Environment: {settings.app_env}")
        console.print(f"Log level: {settings.log_level}")
        console.print(f"LLM Model: {settings.openai_model}")

        return

    try:
        llm = LLMClient(settings)
        response = llm.complete(args.prompt)
    except LLMConfigurationError as error:
        console.print(f"[bold red]Configuration error:[/bold red] {error}")
        raise SystemExit(1)
    except LLMResponseParsingError as error:
        console.print(f"[bold red]Response parsing error:[/bold red] {error}")
        raise SystemExit(1)
    except Exception as error:
        console.print(f"[bold red]LLM call error:[/bold red] {error}")
        raise SystemExit(1)

    parsed = response.parsed

    table = Table(show_header=False, box=None)
    table.add_row("[bold]Answer[/bold]", parsed.answer)
    table.add_row("[bold]Confidence[/bold]", parsed.confidence)
    table.add_row("[bold]Missing context[/bold]",
                  render_items(parsed.missing_context))
    table.add_row("[bold]Next actions[/bold]",
                  render_items(parsed.next_actions))
    table.add_row("[bold]Source references[/bold]",
                  render_items(parsed.source_references))

    console.print(
        Panel.fit(
            table,
            title=f"{response.model} - {response.latency_ms}ms",
            border_style="green"
        )
    )


if __name__ == "__main__":
    main()
