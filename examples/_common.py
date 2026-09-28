"""Command-line flags shared by the examples."""
import argparse


def parse(description, out, epochs, **extra):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-open", action="store_true", help="live server, but don't open a browser tab")
    ap.add_argument("--headless", action="store_true", help="no live server; only write the replay HTML/SVG")
    ap.add_argument("--delay", type=float, default=0.01, help="seconds to sleep per epoch so you can watch")
    ap.add_argument("--epochs", type=int, default=epochs)
    ap.add_argument("--out", default=out)
    for name, (default, help_) in extra.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, action="store_true", help=help_)
        else:
            ap.add_argument(flag, type=type(default), default=default, help=help_)
    return ap.parse_args()


def scope_kwargs(args):
    return {"live": not args.headless, "open_browser": not (args.no_open or args.headless)}


def finish(scope, args):
    print("replay saved to", scope.save_html(args.out))
    print("snapshot saved to", scope.snapshot(args.out.replace(".html", ".svg")))
    if not args.headless:
        scope.wait()
