"""Command line: ``python -m speciesclassifiers <command>`` (or ``speciesclassifiers <command>``).

    app                          open the app in your browser
    new FOLDER --type TYPE       create a project (TYPE: randomforest, calltypes, delphinid); --example uses the example data
    run FOLDER [--stages ...]    run a project (stages already done with the same settings are skipped)
    status FOLDER                show which stages are done
"""

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="speciesclassifiers", description="Build acoustic species classifiers.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("app", help="open the app in your browser")
    a.add_argument("--port", type=int, default=None)
    a.add_argument("--no-browser", action="store_true")
    n = sub.add_parser("new", help="create a project")
    n.add_argument("folder")
    n.add_argument("--type", required=True, choices=["randomforest", "calltypes", "delphinid"])
    n.add_argument("--example", action="store_true", help="point the project at the bundled example data")
    r = sub.add_parser("run", help="run a project")
    r.add_argument("folder")
    r.add_argument("--stages", nargs="*")
    r.add_argument("--force", action="store_true", help="rerun stages even if up to date")
    r.add_argument("--echo", action="store_true", help="also print to the terminal")
    s = sub.add_parser("status", help="show a project's stages")
    s.add_argument("folder")
    args = ap.parse_args(argv)

    if args.cmd == "app":
        from speciesclassifiers.app import main as app_main
        app_main(port=args.port, show=not args.no_browser)
    elif args.cmd == "new":
        from speciesclassifiers.project import WORKFLOWS, Project
        wf = WORKFLOWS[args.type]
        p = Project.create(args.folder, args.type, config=wf.example() if args.example else None)
        print(f"Created {p.folder}")
    elif args.cmd == "run":
        from speciesclassifiers.project import run
        sys.exit(0 if run(args.folder, args.stages, args.force, echo=args.echo) == "finished" else 1)
    elif args.cmd == "status":
        from speciesclassifiers.project import Project
        p = Project(args.folder)
        if not p.exists():
            sys.exit(f"No project in {p.folder}")
        st = p.status()
        for stage in p.workflow.stages:
            state = p.stage_states()[stage.name]
            err = st["stages"].get(stage.name, {}).get("error", "")
            print(f"{state:9s} {stage.title}" + (f"  ({err})" if state == "failed" else ""))


if __name__ == "__main__":
    main()
