import argparse
from .import_cmd import import_pdfs
from .search_cmd import search
from .ask_cmd import ask_question
from .fix_cmd import fix_metadata
from ..indexer.store import get_indexed_paper_ids, remove_paper
from ..rag.session import list_sessions, create_session, delete_session, get_session


def main():
    parser = argparse.ArgumentParser(prog="paper-vault", description="Personal paper reading assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # import
    import_parser = subparsers.add_parser("import", help="Import PDFs and generate notes")
    import_parser.add_argument("paths", nargs="*", help="PDF files or directories (default: $PAPER_VAULT_IMPORT_DIRS or ./papers)")
    import_parser.add_argument("--no-llm", action="store_true", help="Skip LLM note generation")
    import_parser.add_argument("--no-index", action="store_true", help="Skip vector indexing")
    import_parser.add_argument("--force", action="store_true", help="Re-import even if already indexed")

    # search
    search_parser = subparsers.add_parser("search", help="Semantic search across indexed papers")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-k", type=int, default=5, help="Number of results (default: 5)")
    search_parser.add_argument("--year-from", type=int, help="Filter: papers from this year")
    search_parser.add_argument("--year-to", type=int, help="Filter: papers up to this year")
    search_parser.add_argument("--author", type=str, help="Filter: papers by this author (partial match)")

    # ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question about your papers (RAG)")
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument("-n", "--notes", type=int, default=5,
                            help="Max papers to retrieve (default: 5)")
    ask_parser.add_argument("--chunks", type=int, default=None,
                            help="Max detail chunks per paper (default: auto)")
    ask_parser.add_argument("-d", "--detail", type=str, default="auto",
                            choices=["auto", "1", "2", "3", "all"],
                            help="Detail level: auto (LLM judge), 1 (notes only), "
                                 "2 (moderate chunks), 3 (extensive chunks), all (full text)")
    ask_parser.add_argument("--max-tokens", type=int, default=None,
                            help="Max tokens for LLM answer (default: auto — 1024/2048/3072 based on n_papers)")
    ask_parser.add_argument("--year-from", type=int, help="Filter: papers from this year")
    ask_parser.add_argument("--year-to", type=int, help="Filter: papers up to this year")
    ask_parser.add_argument("--author", type=str, help="Filter: papers by this author")
    ask_parser.add_argument("--session", type=str, help="Session ID for multi-turn conversation")
    ask_parser.add_argument("--continue", dest="continue_session", type=str,
                            help="Continue an existing session (alias for --session)")

    # session
    session_parser = subparsers.add_parser("session", help="Manage conversation sessions")
    session_sub = session_parser.add_subparsers(dest="session_cmd", required=True)
    session_sub.add_parser("list", help="List all sessions")
    session_new = session_sub.add_parser("new", help="Create a new session")
    session_new.add_argument("--name", type=str, default="", help="Session name")
    session_delete = session_sub.add_parser("delete", help="Delete a session")
    session_delete.add_argument("session_id", help="Session ID to delete")
    session_show = session_sub.add_parser("show", help="Show session details")
    session_show.add_argument("session_id", help="Session ID to show")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the web UI")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    serve_parser.add_argument("-p", "--port", type=int, default=8080, help="Port (default: 8080)")
    serve_parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")

    # list
    subparsers.add_parser("list", help="List all indexed paper IDs")

    # fix-metadata
    fix_parser = subparsers.add_parser("fix-metadata", help="Re-extract metadata for indexed papers")
    fix_parser.add_argument("paper_ids", nargs="*", help="Paper IDs to fix")
    fix_parser.add_argument("--all", action="store_true", help="Fix all indexed papers")

    # remove
    remove_parser = subparsers.add_parser("remove", help="Remove a paper from the index")
    remove_parser.add_argument("paper_id", help="Paper ID to remove")

    args = parser.parse_args()

    if args.command == "import":
        import_pdfs(args.paths, no_llm=args.no_llm, no_index=args.no_index,
                     force=args.force)
    elif args.command == "search":
        search(args.query, top_k=args.k,
               year_from=args.year_from, year_to=args.year_to, author=args.author)
    elif args.command == "ask":
        session_id = args.session or getattr(args, "continue_session", None)
        ask_question(args.question, n_papers=args.notes, chunks_per_paper=args.chunks,
                     detail=args.detail, max_tokens=args.max_tokens,
                     year_from=args.year_from, year_to=args.year_to, author=args.author,
                     session_id=session_id)
    elif args.command == "serve":
        import uvicorn
        import webbrowser
        import threading
        url = f"http://{args.host}:{args.port}"
        print(f"Paper Vault web UI → {url}")
        if not args.no_open:
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        uvicorn.run("paper_vault.web.app:app", host=args.host, port=args.port)
    elif args.command == "list":
        import lancedb
        from ..config import config
        db = lancedb.connect(str(config.VECTORS_DIR))
        try:
            table = db.open_table("notes_index")
            rows = table.to_arrow().to_pylist()
            print(f"Indexed papers ({len(rows)}):")
            for r in sorted(rows, key=lambda r: r["paper_id"]):
                title = r.get("title", "")[:60] or "-"
                note_file = r.get("note_file", "") or "-"
                print(f"  {r['paper_id']}")
                print(f"    title: {title}")
                print(f"    note:  {note_file}")
        except Exception:
            ids = get_indexed_paper_ids()
            print(f"Indexed papers ({len(ids)}):")
            for pid in sorted(ids):
                print(f"  {pid}")
    elif args.command == "fix-metadata":
        fix_metadata(args.paper_ids, all_papers=args.all)
    elif args.command == "remove":
        if args.paper_id in get_indexed_paper_ids():
            remove_paper(args.paper_id)
            print(f"Removed: {args.paper_id}")
        else:
            print(f"Paper not found: {args.paper_id}")
    elif args.command == "session":
        if args.session_cmd == "list":
            sessions = list_sessions()
            if not sessions:
                print("No sessions.")
            else:
                print(f"Sessions ({len(sessions)}):")
                for s in sessions:
                    name = s["name"] or "(unnamed)"
                    print(f"  {s['id']}  {name}  ({s['turns']} turns, {s['compact_count']} compactions)")
        elif args.session_cmd == "new":
            s = create_session(name=args.name)
            print(f"Created session: {s.id}")
        elif args.session_cmd == "delete":
            if delete_session(args.session_id):
                print(f"Deleted: {args.session_id}")
            else:
                print(f"Session not found: {args.session_id}")
        elif args.session_cmd == "show":
            s = get_session(args.session_id)
            if s:
                print(f"Session: {s.id}")
                print(f"  Name: {s.name or '(unnamed)'}")
                print(f"  Turns: {len(s.turns)}")
                print(f"  Compactions: {s.compact_count}")
                for i, t in enumerate(s.turns):
                    if t.role == "user":
                        q = t.rewritten_question or t.question
                        print(f"  [{i}] Q: {q[:100]}")
                    else:
                        print(f"  [{i}] A: {t.summary[:100]}")
            else:
                print(f"Session not found: {args.session_id}")
